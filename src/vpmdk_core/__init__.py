"""vpmdk: Run machine-learning potentials using VASP style inputs.

The utility consumes VASP-style inputs (POSCAR, INCAR, POTCAR, BCAR) and
executes single-point, relaxation, or molecular dynamics runs with the selected
neural-network potential. Multiple ASE calculators are supported (CHGNet,
M3GNet/MatGL, MACE, MatterSim, Matlantis, Eqnorm, MatRIS, AlphaNet, HIENet,
Nequix, UPET, TACE) and the expected VASP outputs such as CONTCAR and
OUTCAR-style energy logs are produced.
"""

import argparse
import csv
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
import xml.etree.ElementTree as ET
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from pymatgen.io.vasp import Incar, Poscar, Potcar
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from chgnet.model import CHGNet as CHGNetModel
    from chgnet.model import CHGNetCalculator
except Exception:  # pragma: no cover - optional dependency
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
    except Exception:  # pragma: no cover - optional dependency
        MatGLLoadModel = None

    _USING_LEGACY_M3GNET = False
except Exception:  # pragma: no cover - optional dependency
    try:
        from m3gnet.models import M3GNet as LegacyM3GNet  # type: ignore
        from m3gnet.models import M3GNetCalculator  # type: ignore
        from m3gnet.models import Potential as LegacyM3GNetPotential  # type: ignore

        _USING_LEGACY_M3GNET = True
    except Exception:  # pragma: no cover - optional dependency
        M3GNetCalculator = None  # type: ignore
        LegacyM3GNet = None  # type: ignore
        LegacyM3GNetPotential = None  # type: ignore
        _USING_LEGACY_M3GNET = False

try:
    from mace.calculators import MACECalculator
except Exception:  # pragma: no cover - optional dependency
    MACECalculator = None  # type: ignore

try:
    from mattersim.forcefield import MatterSimCalculator
except Exception:  # pragma: no cover - optional dependency
    MatterSimCalculator = None  # type: ignore

try:
    from pfp_api_client.pfp.estimator import Estimator as MatlantisEstimator
    from pfp_api_client.pfp.estimator import EstimatorCalcMode
    from pfp_api_client.pfp.calculators.ase_calculator import (
        ASECalculator as MatlantisASECalculator,
    )
except Exception:  # pragma: no cover - optional dependency
    MatlantisEstimator = None  # type: ignore
    MatlantisASECalculator = None  # type: ignore
    EstimatorCalcMode = None  # type: ignore

try:
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
except Exception:  # pragma: no cover - optional dependency
    ORBCalculator = None  # type: ignore
    ORB_PRETRAINED_MODELS = None  # type: ignore

try:
    from matris.applications.base import MatRISCalculator
    from matris.model.model import MatRIS as MatRISModel
except Exception:  # pragma: no cover - optional dependency
    MatRISCalculator = None  # type: ignore
    MatRISModel = None  # type: ignore

try:
    from eqnorm.calculator import EqnormCalculator
except Exception:  # pragma: no cover - optional dependency
    EqnormCalculator = None  # type: ignore

try:
    from alphanet.infer.calc import AlphaNetCalculator
    from alphanet.config import All_Config as AlphaNetAllConfig
except Exception:  # pragma: no cover - optional dependency
    AlphaNetCalculator = None  # type: ignore
    AlphaNetAllConfig = None  # type: ignore

try:  # pragma: no cover - optional dependency compatibility
    import torch.serialization

    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

try:
    from hienet.hienet_calculator import HIENetCalculator
except Exception:  # pragma: no cover - optional dependency
    HIENetCalculator = None  # type: ignore

try:
    from nequix.calculator import NequixCalculator
except Exception:  # pragma: no cover - optional dependency
    NequixCalculator = None  # type: ignore

try:
    from upet.calculator import UPETCalculator
except Exception:  # pragma: no cover - optional dependency
    UPETCalculator = None  # type: ignore

try:
    from tace.interface.ase import TACEAseCalc
except Exception:  # pragma: no cover - optional dependency
    TACEAseCalc = None  # type: ignore

try:
    from tace.foundations import tace_foundations
except Exception:  # pragma: no cover - optional dependency
    tace_foundations = None  # type: ignore

try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FAIRChemCalculator = None  # type: ignore

FAIRChemV1Calculator = None  # type: ignore
FAIRChemV1Predictor = None  # type: ignore

try:
    from tensorpotential.calculator.asecalculator import TPCalculator
except Exception:  # pragma: no cover - optional dependency
    TPCalculator = None  # type: ignore

try:
    from tensorpotential.calculator.foundation_models import (
        MODELS_NAME_LIST as GRACE_MODEL_NAMES,
        grace_fm,
    )
except Exception:  # pragma: no cover - optional dependency
    GRACE_MODEL_NAMES: List[str] = []
    grace_fm = None  # type: ignore

try:
    from deepmd.calculator import DP as DeePMDCalculator
except Exception:  # pragma: no cover - optional dependency
    DeePMDCalculator = None  # type: ignore

_sevennet_spec = importlib.util.find_spec("sevennet")
if _sevennet_spec is not None:  # pragma: no cover - optional dependency
    try:
        from sevennet.ase import SevenNetCalculator
    except Exception:  # pragma: no cover - handled dynamically
        SevenNetCalculator = None  # type: ignore
else:  # pragma: no cover - optional dependency
    SevenNetCalculator = None  # type: ignore

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.io.lammpsdata import Prism
from ase.io.vasp import write_vasp_xdatcar
from ase.optimize import BFGS
try:
    from ase.constraints import UnitCellFilter, StrainFilter, FixAtoms
except ImportError:  # pragma: no cover - ASE moved filters in newer versions
    from ase.filters import UnitCellFilter, StrainFilter  # type: ignore
    from ase.constraints import FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.md import velocitydistribution

try:
    import resource
except Exception:  # pragma: no cover - optional on non-Unix platforms
    resource = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.andersen import Andersen
except Exception:  # pragma: no cover - handled dynamically
    Andersen = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.langevin import Langevin
except Exception:  # pragma: no cover - handled dynamically
    Langevin = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.bussi import Bussi
except Exception:  # pragma: no cover - handled dynamically
    Bussi = None  # type: ignore

try:  # pragma: no cover - optional thermostat dependency
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
except Exception:  # pragma: no cover - handled dynamically
    NoseHooverChainNVT = None  # type: ignore

_nequip_spec = importlib.util.find_spec("nequip")
_nequip_ase_spec = importlib.util.find_spec("nequip.ase") if _nequip_spec else None
if _nequip_ase_spec is not None:  # pragma: no cover - optional dependency
    from nequip.ase import NequIPCalculator
else:  # pragma: no cover - optional dependency
    NequIPCalculator = None  # type: ignore

DEFAULT_ORB_MODEL = "orb-v3-conservative-20-omat"
DEFAULT_EQNORM_MODEL = "eqnorm-mptrj"
DEFAULT_MATRIS_MODEL = "matris_10m_oam"
_GRAPH_CONVERTER_ALGORITHMS = frozenset({"fast", "legacy"})
DEFAULT_ALPHANET_MODEL = "AlphaNet-MATPES-r2scan"
DEFAULT_HIENET_MODEL = "HIENet-0"
DEFAULT_NEQUIX_MODEL = "nequix-mp-1"
DEFAULT_FAIRCHEM_MODEL = "esen-sm-direct-all-oc25"
DEFAULT_GRACE_MODEL = "GRACE-2L-MP-r6"
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


def _build_nequip_family_calculator(
    bcar_tags: Dict[str, str],
    *,
    require_allegro: bool = False,
    missing_message: str,
):
    """Create NequIP-based calculators that require deployed model files."""

    if require_allegro and importlib.util.find_spec("allegro") is None:
        raise RuntimeError(
            "Allegro calculator not available. Install allegro and dependencies."
        )
    if NequIPCalculator is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    model_name = "Allegro" if require_allegro else "NequIP"
    if not model_path:
        raise ValueError(f"{model_name} requires MODEL pointing to a deployed model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name} model not found: {model_path}")

    device = bcar_tags.get("DEVICE")
    if hasattr(NequIPCalculator, "from_deployed_model"):
        try:
            if device:
                return NequIPCalculator.from_deployed_model(model_path, device=device)
            return NequIPCalculator.from_deployed_model(model_path)
        except Exception as exc:
            ext = os.path.splitext(model_path)[1].lower()
            if ext not in {".pt", ".pth"}:
                raise
            if not hasattr(NequIPCalculator, "from_compiled_model"):
                raise
            try:
                if device:
                    return NequIPCalculator.from_compiled_model(model_path, device=device)
                return NequIPCalculator.from_compiled_model(model_path)
            except Exception as compiled_exc:
                raise compiled_exc from exc

    if hasattr(NequIPCalculator, "from_compiled_model"):
        if device:
            return NequIPCalculator.from_compiled_model(model_path, device=device)
        return NequIPCalculator.from_compiled_model(model_path)

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
        if algorithm in _GRAPH_CONVERTER_ALGORITHMS:
            return algorithm
        supported = ", ".join(sorted(_GRAPH_CONVERTER_ALGORITHMS))
        raise ValueError(
            f"Invalid {tag_name} value: {raw_value!r}. Expected one of: {supported}."
        )
    return None


def _override_model_graph_converter_algorithm(model, *, algorithm: str, backend_name: str):
    """Replace a model's graph converter with the requested algorithm."""

    if algorithm not in _GRAPH_CONVERTER_ALGORITHMS:
        supported = ", ".join(sorted(_GRAPH_CONVERTER_ALGORITHMS))
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
        signature = inspect.signature(type(graph_converter))
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
        if parameter.default is inspect.Signature.empty:
            raise RuntimeError(
                f"{backend_name} graph converter requires {name!r}; cannot set "
                f"{algorithm!r} from the loaded model."
            )

    converter_cls = type(graph_converter)
    module = inspect.getmodule(converter_cls)
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

    actual_algorithm = getattr(new_converter, "algorithm", None)
    if actual_algorithm != algorithm:
        raise RuntimeError(
            f"{backend_name} graph converter requested {algorithm!r} but initialized "
            f"{actual_algorithm!r}."
        )

    model.graph_converter = new_converter
    return model


def _call_with_optional_kwargs(func, /, *args, optional_kwargs: Dict[str, Any] | None = None, **kwargs):
    """Call ``func`` while dropping unsupported optional keyword arguments."""

    filtered_optional_kwargs = {
        key: value for key, value in (optional_kwargs or {}).items() if value is not None
    }
    if not filtered_optional_kwargs:
        return func(*args, **kwargs)

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        try:
            return func(*args, **kwargs, **filtered_optional_kwargs)
        except TypeError:
            return func(*args, **kwargs)

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return func(*args, **kwargs, **filtered_optional_kwargs)

    supported_optional_kwargs = {
        key: value for key, value in filtered_optional_kwargs.items() if key in signature.parameters
    }
    return func(*args, **kwargs, **supported_optional_kwargs)


def _load_chgnet_model(
    *,
    model_path: str | None,
    device: str | None,
    graph_converter_algorithm: str | None,
):
    """Load a CHGNet model with optional graph-converter override."""

    if CHGNetModel is None:
        raise RuntimeError("CHGNet model loader not available. Install chgnet.")

    if model_path and os.path.exists(model_path):
        if graph_converter_algorithm is not None:
            model = CHGNetModel.from_file(model_path)
            return _override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="CHGNet",
            )
        return CHGNetModel.from_file(model_path)

    model = _call_with_optional_kwargs(
        CHGNetModel.load,
        optional_kwargs={"verbose": False, "use_device": device},
    )
    if graph_converter_algorithm is not None:
        model = _override_model_graph_converter_algorithm(
            model,
            algorithm=graph_converter_algorithm,
            backend_name="CHGNet",
        )
    return model


def _build_chgnet_calculator(bcar_tags: Dict[str, str]):
    """Create a CHGNet calculator with optional DEVICE hint."""

    if CHGNetCalculator is None:
        raise RuntimeError("CHGNetCalculator not available. Install chgnet.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))
    graph_converter_algorithm = _resolve_graph_converter_algorithm(
        bcar_tags,
        backend_tag="CHGNET",
    )
    kwargs = {"use_device": device} if device is not None else {}

    if graph_converter_algorithm is not None:
        model = _load_chgnet_model(
            model_path=model_path,
            device=device,
            graph_converter_algorithm=graph_converter_algorithm,
        )
        return CHGNetCalculator(model=model, **kwargs)

    if model_path and os.path.exists(model_path):
        from_file = getattr(CHGNetCalculator, "from_file", None)
        if callable(from_file):
            try:
                return from_file(model_path, **kwargs)
            except TypeError:
                return from_file(model_path)
        try:
            return CHGNetCalculator(model_path, **kwargs)
        except TypeError:
            return CHGNetCalculator(model_path)

    try:
        return CHGNetCalculator(**kwargs)
    except TypeError:
        return CHGNetCalculator()


def _resolve_device(device: str | None) -> str | None:
    """Return user-specified device or best-effort autodetection."""

    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_mace_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create a MACE calculator with optional ``MODEL`` override."""

    if MACECalculator is None:
        raise RuntimeError("MACECalculator not available. Install mace-torch and dependencies.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))

    if model_path and os.path.exists(model_path):
        return MACECalculator(model_path, device=device)
    return MACECalculator(device=device)


def _build_m3gnet_calculator(bcar_tags: Dict[str, str]):
    """Create a MatGL or legacy M3GNet calculator based on availability."""

    if M3GNetCalculator is None:
        raise RuntimeError("M3GNetCalculator not available. Install matgl or m3gnet.")

    model_path = bcar_tags.get("MODEL")
    device = _resolve_device(bcar_tags.get("DEVICE"))

    if not _USING_LEGACY_M3GNET:
        kwargs = {"device": device} if device is not None else {}
        if model_path and os.path.exists(model_path):
            if MatGLLoadModel is not None:
                try:
                    potential = MatGLLoadModel(model_path)
                    return M3GNetCalculator(potential, **kwargs)
                except Exception:
                    pass
            try:
                return M3GNetCalculator(model_path, **kwargs)
            except TypeError:
                return M3GNetCalculator(model_path)
        try:
            return M3GNetCalculator(**kwargs)
        except TypeError:
            return M3GNetCalculator()

    potential = None
    if model_path and os.path.exists(model_path) and LegacyM3GNetPotential is not None:
        try:
            potential = LegacyM3GNetPotential.from_checkpoint(model_path)
        except Exception:
            try:
                if LegacyM3GNet is not None:
                    potential = LegacyM3GNetPotential(
                        LegacyM3GNet.load(model_path)  # type: ignore[arg-type]
                    )
            except Exception:
                potential = None

    if (
        potential is None
        and LegacyM3GNetPotential is not None
        and LegacyM3GNet is not None
    ):
        potential = LegacyM3GNetPotential(LegacyM3GNet.load())

    if potential is None:
        raise RuntimeError("Legacy M3GNet calculator could not be initialized from available models.")

    if device is not None:
        try:
            return M3GNetCalculator(potential=potential, device=device)
        except TypeError:
            pass

    return M3GNetCalculator(potential=potential)


def _build_simple_model_calculator(
    calculator_cls,
    bcar_tags: Dict[str, str],
    missing_message: str,
):
    """Return calculator initialized with optional ``MODEL`` path."""

    if calculator_cls is None:
        raise RuntimeError(missing_message)

    model_path = bcar_tags.get("MODEL")
    if model_path and os.path.exists(model_path):
        return calculator_cls(model_path)
    return calculator_cls()


def parse_key_value_file(path: str) -> Dict[str, str]:
    """Parse simple key=value style file."""
    data: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            for comment in ("#", "!"):
                if comment in line:
                    line = line.split(comment, 1)[0]
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            data[k.strip().upper()] = v.strip()
    # Backward compatibility: interpret legacy NNP tag as MLP when MLP is absent.
    if "MLP" not in data and "NNP" in data:
        data["MLP"] = data["NNP"]
    return data


def _resolve_mlp_tag(bcar_tags: Dict[str, str], *, default: str = "CHGNET") -> str:
    """Return selected BCAR potential tag using ``MLP`` with legacy ``NNP`` fallback."""

    if "MLP" in bcar_tags:
        mlp_value = str(bcar_tags.get("MLP", "")).strip()
        if not mlp_value:
            raise ValueError("BCAR tag MLP is present but empty.")
        return mlp_value.upper()

    if "NNP" in bcar_tags:
        nnp_value = str(bcar_tags.get("NNP", "")).strip()
        if not nnp_value:
            raise ValueError("BCAR tag NNP is present but empty.")
        return nnp_value.upper()

    return default.strip().upper()


def _flatten(values: Iterable[object]) -> List[float]:
    """Return flattened list of floats from nested sequences."""

    flattened: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flattened.extend(_flatten(item))
        else:
            try:
                flattened.append(float(item))
            except (TypeError, ValueError):
                continue
    return flattened


def _parse_magmom_values(value) -> List[float]:
    """Parse VASP-style MAGMOM definition into a list of floats."""

    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return _flatten(value)

    text = str(value).strip()
    if not text:
        return []

    tokens = text.replace(",", " ").split()
    result: List[float] = []
    for token in tokens:
        if not token:
            continue
        if "*" in token:
            count_str, moment_str = token.split("*", 1)
            try:
                count = int(float(count_str))
            except (TypeError, ValueError):
                continue
            nested = _parse_magmom_values(moment_str)
            if not nested:
                try:
                    nested = [float(moment_str)]
                except (TypeError, ValueError):
                    continue
            if len(nested) == 1:
                result.extend(nested * count)
            else:
                for _ in range(count):
                    result.extend(nested)
            continue
        try:
            result.append(float(token))
        except (TypeError, ValueError):
            continue
    return result


def _normalize_species_labels(symbols: Iterable[object]) -> List[str]:
    """Return species labels with POTCAR-style suffixes removed."""

    normalized: List[str] = []
    for symbol in symbols:
        text: str = ""
        if isinstance(symbol, str):
            text = symbol.strip()
        elif hasattr(symbol, "symbol"):
            text = str(getattr(symbol, "symbol", "")).strip()
        else:
            try:
                text = str(symbol).strip()
            except Exception:
                continue
        if not text:
            continue
        base = text.split("_", 1)[0].strip()
        normalized.append(base or text)
    return normalized


def _infer_type_map(structure) -> List[str]:
    """Infer a DeePMD type map from the provided structure when possible."""

    labels: List[str] = []
    for attr in ("site_symbols", "species"):
        symbols = getattr(structure, attr, None)
        if symbols:
            labels = _normalize_species_labels(symbols)
            if labels:
                break

    unique: List[str] = []
    for label in labels:
        if label and label not in unique:
            unique.append(label)

    return unique


def _expand_magmom_to_atoms(magmoms: List[float], atoms) -> List[float] | None:
    """Expand species MAGMOM values to per-atom list when necessary."""

    if not magmoms:
        return None

    num_atoms = len(atoms)
    if len(magmoms) == num_atoms:
        return magmoms

    symbols = atoms.get_chemical_symbols()
    species_counts: List[int] = []
    previous_symbol: str | None = None
    for symbol in symbols:
        if symbol == previous_symbol:
            species_counts[-1] += 1
        else:
            species_counts.append(1)
            previous_symbol = symbol

    if len(magmoms) == len(species_counts):
        expanded: List[float] = []
        for moment, count in zip(magmoms, species_counts):
            expanded.extend([moment] * count)
        return expanded

    return None


def _apply_initial_magnetization(atoms, incar) -> None:
    """Populate initial magnetic moments from INCAR when available."""

    if not hasattr(incar, "get"):
        return
    if "MAGMOM" not in incar:
        return

    raw = incar.get("MAGMOM")
    magmoms = _parse_magmom_values(raw)
    if not magmoms:
        return
    expanded = _expand_magmom_to_atoms(magmoms, atoms)
    if expanded is None or len(expanded) != len(atoms):
        print(
            "Warning: Unable to reconcile MAGMOM values with number of atoms; "
            "initial magnetic moments will not be set."
        )
        return
    atoms.set_initial_magnetic_moments(expanded)


def _append_xdatcar_configuration(path: str, atoms, frame_number: int) -> None:
    """Append a single XDATCAR configuration block for ``atoms``."""

    scaled_positions = atoms.get_scaled_positions()
    float_string = "{:11.8f}"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"Direct configuration={frame_number:6d}\n")
        for row in scaled_positions:
            handle.write(" ")
            handle.write(" ".join(float_string.format(value) for value in row))
            handle.write("\n")


def _count_symbols_in_order(atoms) -> tuple[list[str], list[int]]:
    """Return unique symbols in order of appearance with their counts."""

    counts: OrderedDict[str, int] = OrderedDict()
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return list(counts.keys()), list(counts.values())


_XDATCAR_STATE: Dict[str, Dict[str, Any]] = {}


def _initialize_xdatcar_state(path: str, atoms) -> None:
    """Capture header metadata needed for XDATCAR appends."""

    symbols, counts = _count_symbols_in_order(atoms)
    comment = atoms.info.get("comment", "Generated by ASE")
    scaling = "1.0"
    species_line = " ".join(symbols)
    counts_line = " ".join(str(value) for value in counts)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            header_lines = handle.readlines()
    except FileNotFoundError:
        header_lines = []

    if len(header_lines) >= 1:
        comment = header_lines[0].rstrip("\n")
    if len(header_lines) >= 2:
        scaling = header_lines[1].strip()
    if len(header_lines) >= 6:
        species_line = header_lines[5].strip()
    if len(header_lines) >= 7:
        counts_line = header_lines[6].strip()

    _XDATCAR_STATE[path] = {
        "initial_cell": atoms.get_cell().array.copy(),
        "previous_cell": atoms.get_cell().array.copy(),
        "comment": comment,
        "scaling": scaling,
        "species_line": species_line,
        "counts_line": counts_line,
        "variable_cell": False,
    }


def _append_variable_cell_configuration(path: str, atoms, frame_number: int) -> None:
    """Append a POSCAR-style block when the lattice changes during MD."""

    state = _XDATCAR_STATE.get(path)
    if state is None:
        _initialize_xdatcar_state(path, atoms)
        state = _XDATCAR_STATE[path]

    float_string = "{:11.8f}"
    cell_string = "{:16.10f}"
    scaled_positions = atoms.get_scaled_positions()

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{state['comment']}\n")
        handle.write(f"{state['scaling']}\n")
        for vector in atoms.get_cell().array:
            handle.write(" ".join(cell_string.format(value) for value in vector))
            handle.write("\n")
        handle.write(f"{state['species_line']}\n")
        handle.write(f"{state['counts_line']}\n")
        handle.write(f"Direct configuration={frame_number:6d}\n")
        for row in scaled_positions:
            handle.write(" ")
            handle.write(" ".join(float_string.format(value) for value in row))
            handle.write("\n")


def _rewrite_first_xdatcar_frame(path: str, atoms) -> None:
    """Ensure the first XDATCAR frame uses direct (fractional) coordinates."""

    scaled_positions = atoms.get_scaled_positions()
    float_string = "{:11.8f}"
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    configuration_index: int | None = None
    for index, line in enumerate(lines):
        if "configuration=" in line.lower():
            configuration_index = index
            break

    if configuration_index is None:
        return

    lines[configuration_index] = f"Direct configuration={1:6d}\n"
    start = configuration_index + 1
    formatted_positions = [
        " " + " ".join(float_string.format(value) for value in row) + "\n"
        for row in scaled_positions
    ]
    end = start + len(formatted_positions)
    lines[start:end] = formatted_positions

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def _write_xdatcar_step(path: str, atoms, step_index: int) -> None:
    """Write or append an XDATCAR snapshot for the given MD ``step_index``."""

    frame_number = step_index + 1
    if step_index == 0:
        with open(path, "w", encoding="utf-8") as handle:
            write_vasp_xdatcar(handle, [atoms])
        _rewrite_first_xdatcar_frame(path, atoms)
        _initialize_xdatcar_state(path, atoms)
        return

    state = _XDATCAR_STATE.get(path)
    if state is None:
        _initialize_xdatcar_state(path, atoms)
        state = _XDATCAR_STATE[path]

    cell_changed = not np.allclose(
        atoms.get_cell().array, state["previous_cell"], rtol=1e-10, atol=1e-12
    )
    state["previous_cell"] = atoms.get_cell().array.copy()

    if state["variable_cell"] or cell_changed:
        state["variable_cell"] = True
        _append_variable_cell_configuration(path, atoms, frame_number)
        return

    _append_xdatcar_configuration(path, atoms, frame_number)


def _write_lammps_trajectory_step(path: str, atoms, step_index: int) -> None:
    """Write or append a LAMMPS trajectory frame for the given MD step."""

    append = step_index != 0
    file_mode = "a" if append else "w"

    prism = Prism(atoms.get_cell().array, atoms.get_pbc())
    lx, ly, lz, xy, xz, yz = prism.get_lammps_prism()
    # Convert the prism representation (box lengths and tilt factors) into the
    # bounds expected by the LAMMPS dump format. See "How a triclinic box is
    # defined" in the LAMMPS documentation for the bound transformation.
    x_tilt_min = min(0.0, xy, xz, xy + xz)
    x_tilt_max = max(0.0, xy, xz, xy + xz)
    xlo = 0.0 - x_tilt_min
    xhi = lx - x_tilt_max
    y_tilt_min = min(0.0, yz)
    y_tilt_max = max(0.0, yz)
    ylo = 0.0 - y_tilt_min
    yhi = ly - y_tilt_max
    zlo = 0.0
    zhi = lz
    pbc_flags = ["pp" if periodic else "ff" for periodic in atoms.get_pbc()]

    species_to_type: Dict[str, int] = {}
    symbols = atoms.get_chemical_symbols()
    for symbol in symbols:
        if symbol not in species_to_type:
            species_to_type[symbol] = len(species_to_type) + 1

    # Obtain atomic positions in the LAMMPS coordinate system without wrapping
    lammps_positions = prism.vector_to_lammps(atoms.get_positions(), wrap=False)
    cell_matrix = prism.cell

    # Convert to fractional coordinates in the LAMMPS cell and extract image flags
    fractional = np.linalg.solve(cell_matrix.T, lammps_positions.T).T
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    image_flags = (np.floor(fractional).astype(int)) * pbc
    scaled_positions = fractional - image_flags

    velocities = atoms.get_velocities()
    velocity_data = None
    if velocities is not None:
        velocity_data = prism.vector_to_lammps(velocities, wrap=False)

    with open(path, file_mode, encoding="utf-8") as handle:
        handle.write("ITEM: TIMESTEP\n")
        handle.write(f"{step_index + 1}\n")
        handle.write("ITEM: NUMBER OF ATOMS\n")
        handle.write(f"{len(atoms)}\n")
        handle.write(
            "ITEM: BOX BOUNDS xy xz yz " + " ".join(pbc_flags) + "\n"
        )
        handle.write(f"{xlo} {xhi} {xy}\n")
        handle.write(f"{ylo} {yhi} {xz}\n")
        handle.write(f"{zlo} {zhi} {yz}\n")

        columns = ["id", "type", "xs", "ys", "zs", "ix", "iy", "iz"]
        if velocity_data is not None:
            columns.extend(["vx", "vy", "vz"])
        handle.write("ITEM: ATOMS " + " ".join(columns) + "\n")

        for index, (scaled, images, symbol) in enumerate(
            zip(scaled_positions, image_flags, symbols), start=1
        ):
            type_id = species_to_type[symbol]
            values = [index, type_id, *scaled.tolist(), *images.tolist()]
            if velocity_data is not None:
                values.extend(velocity_data[index - 1].tolist())
            handle.write(" ".join(str(value) for value in values) + "\n")


@dataclass
class _VasprunStep:
    """Container for one ionic step written to ``vasprun.xml``."""

    cell: list[list[float]]
    scaled_positions: list[list[float]]
    forces: list[list[float]]
    stress: list[list[float]] | None
    potential_energy: float
    total_energy: float
    kinetic_energy: float
    thermostat_potential: float
    thermostat_kinetic: float
    temperature: float
    sc_time: float = 0.0


@dataclass(frozen=True)
class _PseudoScfSettings:
    """Pseudo electronic-step settings used for VASP-compatibility output."""

    enabled: bool = False
    nelm: int = 60
    nelmin: int = 2
    nelmdl: int = 0
    ediff: float = 1.0e-4


_PSEUDO_SCF_INCAR_TAGS = frozenset({"NELM", "NELMIN", "NELMDL", "EDIFF"})
_ACTIVE_PSEUDO_SCF_SETTINGS: _PseudoScfSettings | None = None


@dataclass(frozen=True)
class _VaspInputPaths:
    """Selected run input paths reused by compatibility writers."""

    incar_path: str | None = None
    potcar_path: str | None = None
    kpoints_path: str | None = None


_ACTIVE_VASP_INPUT_PATHS: _VaspInputPaths | None = None


@dataclass
class _VaspCompatRecorder:
    """State tracker for VASP-like ``OUTCAR``/``OSZICAR``/``vasprun.xml`` output."""

    symbols: List[str]
    initial_cell: list[list[float]]
    initial_scaled_positions: list[list[float]]
    ibrion: int
    potim: float | None
    mdalgo: int | None
    isif: int | None = None
    stress_mode: str = "none"
    neb_mode: bool = False
    pseudo_scf: _PseudoScfSettings = field(default_factory=_PseudoScfSettings)
    oszicar_scf_header_written: bool = False
    neb_prev_positions: np.ndarray | None = None
    neb_next_positions: np.ndarray | None = None
    started_at: float = field(default_factory=time.perf_counter)
    previous_energy: float | None = None
    steps: List[_VasprunStep] = field(default_factory=list)


@dataclass
class _NebImageResult:
    """Final-step summary extracted from one NEB image directory."""

    image_name: str
    atoms: Any
    potential_energy: float
    forces: np.ndarray
    stress: np.ndarray | None


@dataclass
class _NebChainApproximation:
    """Approximate NEB chain components derived from neighboring images."""

    tangential_force: float
    tangent_vectors: np.ndarray
    chain_force_vectors: np.ndarray
    chain_plus_total: np.ndarray


def _coerce_neb_reference_positions(values) -> np.ndarray | None:
    """Return neighbor image positions as ``(n_atoms, 3)`` array when valid."""

    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[1] != 3:
        return None
    return np.array(array, dtype=float, copy=True)


def _matrix_to_nested_list(values) -> list[list[float]]:
    """Return ``values`` as nested Python ``float`` lists."""

    return np.asarray(values, dtype=float).tolist()


def _safe_get_forces(atoms) -> np.ndarray:
    """Return per-atom forces or zeros when unavailable."""

    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        return np.zeros((len(atoms), 3), dtype=float)


def _stress_mode_from_isif(isif: int | None) -> str:
    """Return stress output mode from VASP ``ISIF`` semantics."""

    if isif is None:
        return "none"
    if isif <= 0:
        return "none"
    if isif == 1:
        return "trace"
    return "full"


def _voigt_to_full_stress(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert ASE Voigt stress ``[xx, yy, zz, yz, xz, xy]`` to 3x3 matrix."""

    xx, yy, zz, yz, xz, xy = [float(v) for v in stress_voigt]
    return np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ],
        dtype=float,
    )


def _full_to_voigt_stress(stress_matrix: np.ndarray) -> np.ndarray:
    """Convert full 3x3 stress matrix to ASE Voigt convention."""

    return np.array(
        [
            float(stress_matrix[0, 0]),
            float(stress_matrix[1, 1]),
            float(stress_matrix[2, 2]),
            float(stress_matrix[1, 2]),
            float(stress_matrix[0, 2]),
            float(stress_matrix[0, 1]),
        ],
        dtype=float,
    )


def _safe_get_stress_matrix(atoms, *, mode: str) -> np.ndarray | None:
    """Return stress matrix in eV/A^3 based on output mode."""

    if mode == "none":
        return None

    try:
        raw = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        return None

    if raw.shape == (6,):
        stress_voigt = raw
    elif raw.shape == (3, 3):
        stress_voigt = _full_to_voigt_stress(raw)
    else:
        return None

    if mode == "trace":
        mean_pressure = float(np.mean(stress_voigt[:3]))
        stress_voigt = np.array([mean_pressure, mean_pressure, mean_pressure, 0.0, 0.0, 0.0])

    return _voigt_to_full_stress(stress_voigt)


def _estimate_neb_chain_approximation(
    *,
    positions: np.ndarray,
    forces: np.ndarray,
    prev_positions: np.ndarray | None,
    next_positions: np.ndarray | None,
) -> _NebChainApproximation | None:
    """Estimate NEB chain vectors from neighboring image displacements."""

    if positions.shape != forces.shape or positions.ndim != 2 or positions.shape[1] != 3:
        return None

    prev = prev_positions if prev_positions is not None and prev_positions.shape == positions.shape else None
    nxt = next_positions if next_positions is not None and next_positions.shape == positions.shape else None

    if prev is not None and nxt is not None:
        tangent_raw = nxt - prev
    elif nxt is not None:
        tangent_raw = nxt - positions
    elif prev is not None:
        tangent_raw = positions - prev
    else:
        tangent_raw = np.zeros_like(forces)

    tangent_norm = float(np.linalg.norm(tangent_raw.ravel()))
    if tangent_norm <= 1e-14:
        tangent_vectors = np.zeros_like(forces)
        tangential_force = 0.0
        chain_force_vectors = np.zeros_like(forces)
    else:
        tangent_vectors = tangent_raw / tangent_norm
        tangential_force = float(np.dot(forces.ravel(), tangent_vectors.ravel()))
        chain_force_vectors = tangent_vectors * tangential_force

    if forces.size:
        chain_plus_total = np.sum(chain_force_vectors + forces, axis=0)
    else:
        chain_plus_total = np.zeros(3, dtype=float)

    return _NebChainApproximation(
        tangential_force=tangential_force,
        tangent_vectors=tangent_vectors,
        chain_force_vectors=chain_force_vectors,
        chain_plus_total=chain_plus_total,
    )


def _read_non_comment_lines(path: str) -> list[str]:
    """Return stripped non-empty lines with ``#``/``!`` comments removed."""

    if not os.path.exists(path):
        return []
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            for marker in ("#", "!"):
                if marker in line:
                    line = line.split(marker, 1)[0]
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _extract_potcar_titles(path: str) -> list[str]:
    """Return POTCAR TITEL strings from ``path`` when available."""

    if not os.path.exists(path):
        return []
    titles: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "TITEL" not in line or "=" not in line:
                continue
            title = line.split("=", 1)[1].strip()
            if title:
                titles.append(title)
    return titles


def _append_outcar_metadata_header(handle, atoms) -> None:
    """Append VASP-like metadata/header blocks to ``OUTCAR``."""

    paths = _ACTIVE_VASP_INPUT_PATHS or _VaspInputPaths("INCAR", "POTCAR", "KPOINTS")
    incar_lines = _read_non_comment_lines(paths.incar_path) if paths.incar_path else []
    if incar_lines:
        handle.write(" INCAR:\n")
        for line in incar_lines:
            handle.write(f"   {line}\n")

    potcar_titles = _extract_potcar_titles(paths.potcar_path) if paths.potcar_path else []
    if potcar_titles:
        for title in potcar_titles:
            handle.write(f" POTCAR:    {title}\n")
    elif len(atoms):
        seen: OrderedDict[str, None] = OrderedDict()
        for symbol in atoms.get_chemical_symbols():
            seen.setdefault(symbol, None)
        for symbol in seen:
            handle.write(f" POTCAR:    PAW_PBE {symbol}\n")

    cell = np.asarray(atoms.get_cell().array, dtype=float)
    if abs(np.linalg.det(cell)) > 1e-14:
        reciprocal = np.linalg.inv(cell).T
    else:
        reciprocal = np.zeros((3, 3), dtype=float)
    handle.write("      direct lattice vectors                 reciprocal lattice vectors\n")
    for direct, recip in zip(cell, reciprocal):
        handle.write(
            f"  {direct[0]:12.9f} {direct[1]:12.9f} {direct[2]:12.9f}"
            f"   {recip[0]:12.9f} {recip[1]:12.9f} {recip[2]:12.9f}\n"
        )

    kp_lines = _read_non_comment_lines(paths.kpoints_path) if paths.kpoints_path else []
    kpoint_label = "Gamma"
    if len(kp_lines) >= 3:
        kpoint_label = kp_lines[2]
    handle.write(
        f" k-points in reciprocal lattice and weights: {kpoint_label:<40}\n"
    )
    handle.write("   0.00000000   0.00000000   0.00000000      1.00000000\n\n")


def _append_kpoints_xml(parent) -> None:
    """Append a minimal Gamma-only ``kpoints`` section."""

    kpoints = ET.SubElement(parent, "kpoints")
    ET.SubElement(kpoints, "generation", {"param": "Gamma"})
    kpointlist = ET.SubElement(kpoints, "varray", {"name": "kpointlist"})
    ET.SubElement(kpointlist, "v").text = "       0.00000000       0.00000000       0.00000000 "
    weights = ET.SubElement(kpoints, "varray", {"name": "weights"})
    ET.SubElement(weights, "v").text = "       1.00000000 "


def _pseudo_scf_settings_from_incar(incar, *, enabled: bool) -> _PseudoScfSettings:
    """Return pseudo-SCF settings derived from the selected run ``INCAR``."""

    if not enabled:
        return _PseudoScfSettings(enabled=False)

    if not hasattr(incar, "get"):
        return _PseudoScfSettings(enabled=enabled)

    def _parse_int_tag(key: str, default: int) -> int:
        raw = incar.get(key, default)
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default

    def _parse_float_tag(key: str, default: float) -> float:
        raw = incar.get(key, default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    nelm = max(1, _parse_int_tag("NELM", 60))
    nelmin = min(max(1, _parse_int_tag("NELMIN", 2)), nelm)
    nelmdl = _parse_int_tag("NELMDL", 0)
    ediff = max(_parse_float_tag("EDIFF", 1.0e-4), 0.0)
    return _PseudoScfSettings(
        enabled=enabled,
        nelm=nelm,
        nelmin=nelmin,
        nelmdl=nelmdl,
        ediff=ediff,
    )


@contextmanager
def _active_pseudo_scf_settings(settings: _PseudoScfSettings):
    """Temporarily expose pseudo-SCF settings to nested output writers."""

    global _ACTIVE_PSEUDO_SCF_SETTINGS
    previous = _ACTIVE_PSEUDO_SCF_SETTINGS
    _ACTIVE_PSEUDO_SCF_SETTINGS = settings
    try:
        yield
    finally:
        _ACTIVE_PSEUDO_SCF_SETTINGS = previous


@contextmanager
def _active_vasp_input_paths(paths: _VaspInputPaths):
    """Temporarily expose selected run input paths to compatibility writers."""

    global _ACTIVE_VASP_INPUT_PATHS
    previous = _ACTIVE_VASP_INPUT_PATHS
    _ACTIVE_VASP_INPUT_PATHS = paths
    try:
        yield
    finally:
        _ACTIVE_VASP_INPUT_PATHS = previous


def _selected_incar_path() -> str:
    """Return the active run ``INCAR`` path or the caller's local ``INCAR``."""

    paths = _ACTIVE_VASP_INPUT_PATHS or _VaspInputPaths()
    return paths.incar_path or "INCAR"


def _resolve_pseudo_scf_settings(*, enabled: bool) -> _PseudoScfSettings:
    """Return pseudo-SCF settings from the active run or selected ``INCAR``."""

    if _ACTIVE_PSEUDO_SCF_SETTINGS is not None:
        active = _ACTIVE_PSEUDO_SCF_SETTINGS
        return _PseudoScfSettings(
            enabled=enabled,
            nelm=active.nelm,
            nelmin=active.nelmin,
            nelmdl=active.nelmdl,
            ediff=active.ediff,
        )
    if not enabled:
        return _PseudoScfSettings(enabled=False)
    return _pseudo_scf_settings_from_incar(_load_incar(_selected_incar_path()), enabled=True)


def _format_outcar_ediff(value: float) -> str:
    """Return VASP-like scientific notation for ``EDIFF`` lines in ``OUTCAR``."""

    if value == 0.0:
        return "0.0E+00"
    mantissa_text, exponent_text = f"{value:.8E}".split("E")
    digits = mantissa_text.replace(".", "").rstrip("0")
    exponent = int(exponent_text) + 1
    return f"0.{digits or '0'}E{exponent:+03d}"


def _initialize_vasp_compat_outputs(
    atoms,
    *,
    ibrion: int,
    potim: float | None = None,
    mdalgo: int | None = None,
    isif: int | None = None,
    neb_mode: bool = False,
    write_oszicar_pseudo_scf: bool = False,
    neb_prev_positions: np.ndarray | None = None,
    neb_next_positions: np.ndarray | None = None,
) -> _VaspCompatRecorder:
    """Initialize compatibility outputs and return recorder state."""

    pseudo_scf = _resolve_pseudo_scf_settings(enabled=write_oszicar_pseudo_scf)
    initial_cell = _matrix_to_nested_list(atoms.get_cell().array)
    initial_scaled_positions = _matrix_to_nested_list(atoms.get_scaled_positions())
    current_positions = np.asarray(atoms.get_positions(), dtype=float)
    prev_positions = _coerce_neb_reference_positions(neb_prev_positions)
    if prev_positions is not None and prev_positions.shape != current_positions.shape:
        prev_positions = None
    next_positions = _coerce_neb_reference_positions(neb_next_positions)
    if next_positions is not None and next_positions.shape != current_positions.shape:
        next_positions = None
    recorder = _VaspCompatRecorder(
        symbols=list(atoms.get_chemical_symbols()),
        initial_cell=initial_cell,
        initial_scaled_positions=initial_scaled_positions,
        ibrion=ibrion,
        potim=potim,
        mdalgo=mdalgo,
        isif=isif,
        stress_mode=_stress_mode_from_isif(isif),
        neb_mode=neb_mode,
        pseudo_scf=pseudo_scf,
        neb_prev_positions=prev_positions,
        neb_next_positions=next_positions,
    )

    with open("OUTCAR", "w", encoding="utf-8") as handle:
        handle.write(" vasp.6.x compatible output generated by VPMDK\n")
        handle.write(f"   IBRION = {ibrion:6d}\n")
        if isif is not None:
            handle.write(f"   ISIF   = {isif:6d}\n")
        if mdalgo is not None:
            handle.write(f"   MDALGO = {mdalgo:6d}\n")
        if potim is not None:
            handle.write(f"   POTIM  = {potim:6.4f}    time-step for ionic-motion\n")
        if pseudo_scf.enabled:
            handle.write(" Electronic Relaxation 1\n")
            handle.write(
                f"   NELM   = {pseudo_scf.nelm:6d};   NELMIN={pseudo_scf.nelmin:3d};"
                f" NELMDL={pseudo_scf.nelmdl:3d}     # of ELM steps \n"
            )
            handle.write(
                f"   EDIFF  = {_format_outcar_ediff(pseudo_scf.ediff):>10s}"
                "   stopping-criterion for ELM\n"
            )
        handle.write(f"   ICHAIN = {0:6d}\n")
        handle.write(
            f"   number of dos      NEDOS =    301   number of ions     NIONS = {len(atoms):6d}\n"
        )
        handle.write("\n")
        _append_outcar_metadata_header(handle, atoms)

    with open("OSZICAR", "w", encoding="utf-8"):
        pass

    return recorder


def _append_outcar_compat_step(
    step_index: int,
    atoms,
    forces: np.ndarray,
    stress_matrix: np.ndarray | None,
    pseudo_scf: _PseudoScfSettings,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float,
    thermostat_potential: float,
    thermostat_kinetic: float,
    neb_mode: bool = False,
    neb_chain: _NebChainApproximation | None = None,
) -> None:
    """Append a VTST-friendly ionic-step block to ``OUTCAR``."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if forces.size:
        norms = np.linalg.norm(forces, axis=1)
        force_max = float(np.max(norms))
        force_rms = float(np.sqrt(np.mean(norms * norms)))
    else:
        force_max = 0.0
        force_rms = 0.0

    with open("OUTCAR", "a", encoding="utf-8") as handle:
        handle.write(
            f"\n--------------------------------------- Ionic step {step_index:8d}  -------------------------------------------\n\n"
        )
        handle.write(
            f"\n--------------------------------------- Iteration {step_index:6d}(   1)  ---------------------------------------\n"
        )
        handle.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
        handle.write(" -----------------------------------------------------------------------------------\n")
        for position, force in zip(positions, forces):
            handle.write(
                f" {position[0]:16.8f} {position[1]:16.8f} {position[2]:16.8f}"
                f" {force[0]:16.8f} {force[1]:16.8f} {force[2]:16.8f}\n"
            )
        handle.write(" -----------------------------------------------------------------------------------\n")
        drift = np.sum(forces, axis=0) if forces.size else np.zeros(3, dtype=float)
        handle.write(
            "    total drift:                               "
            f"{drift[0]:12.6f} {drift[1]:12.6f} {drift[2]:12.6f}\n\n"
        )
        handle.write(f" FORCES: max atom, RMS {force_max:16.8f} {force_rms:16.8f}\n\n")
        if stress_matrix is not None:
            xx = float(stress_matrix[0, 0])
            yy = float(stress_matrix[1, 1])
            zz = float(stress_matrix[2, 2])
            xy = float(stress_matrix[0, 1])
            yz = float(stress_matrix[1, 2])
            zx = float(stress_matrix[2, 0])
            to_kbar = 1.0 / KBAR_TO_EV_PER_A3
            ext_pressure = (xx + yy + zz) / 3.0 * to_kbar
            handle.write("  FORCE on cell =-STRESS in cart. coord.  units (eV):\n")
            handle.write("  Direction    XX          YY          ZZ          XY          YZ          ZX\n")
            handle.write("  -------------------------------------------------------------------------------------\n")
            handle.write(
                f"  Total   {xx:11.5f} {yy:11.5f} {zz:11.5f}"
                f" {xy:11.5f} {yz:11.5f} {zx:11.5f}\n"
            )
            handle.write(
                f"  in kB   {xx * to_kbar:11.2f} {yy * to_kbar:11.2f} {zz * to_kbar:11.2f}"
                f" {xy * to_kbar:11.2f} {yz * to_kbar:11.2f} {zx * to_kbar:11.2f}\n"
            )
            handle.write(
                f"  external pressure = {ext_pressure:11.2f} kB  Pullay stress =        0.00 kB\n\n"
            )
        if neb_mode:
            if neb_chain is None:
                tangential_force = 0.0
                tangent_vectors = np.zeros_like(forces)
                chain_force_vectors = np.zeros_like(forces)
                chain_plus_total = np.zeros(3, dtype=float)
            else:
                tangential_force = float(neb_chain.tangential_force)
                tangent_vectors = np.asarray(neb_chain.tangent_vectors, dtype=float)
                if tangent_vectors.shape != forces.shape:
                    tangent_vectors = np.zeros_like(forces)
                chain_force_vectors = np.asarray(neb_chain.chain_force_vectors, dtype=float)
                if chain_force_vectors.shape != forces.shape:
                    chain_force_vectors = np.zeros_like(forces)
                chain_plus_total = np.asarray(neb_chain.chain_plus_total, dtype=float)
                if chain_plus_total.shape != (3,):
                    chain_plus_total = np.zeros(3, dtype=float)

            perpendicular_forces = forces - chain_force_vectors
            if perpendicular_forces.size:
                chain_force_max = float(np.max(np.linalg.norm(perpendicular_forces, axis=1)))
            else:
                chain_force_max = 0.0

            handle.write(
                " NEB: projections on to tangent (spring, REAL) "
                f"{0.0:12.6f} {tangential_force:12.6f} {chain_force_max:12.6f}\n\n"
            )
        handle.write("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n")
        handle.write("  ---------------------------------------------------\n")
        handle.write(f"  free  energy   TOTEN  = {potential_energy:16.8f} eV\n")
        handle.write(
            f"  energy  without entropy= {potential_energy:16.8f}  "
            f"energy(sigma->0) = {potential_energy:16.8f}\n"
        )
        handle.write(f"  kinetic energy EKIN    = {kinetic_energy:16.8f} eV\n")
        handle.write(f"  nose potential         = {thermostat_potential:16.8f} eV\n")
        handle.write(f"  nose kinetic           = {thermostat_kinetic:16.8f} eV\n")
        handle.write(f"  total energy ETOTAL    = {total_energy:16.8f} eV\n\n")
        if neb_mode:
            handle.write(f"  tangential force (eV/A) {tangential_force:16.6f}\n")
            handle.write(" TANGENT                                        CHAIN-FORCE (eV/Angst)\n")
            handle.write(" -------------------------------------------------------------------------------\n")
            for tangent, chain_force in zip(tangent_vectors, chain_force_vectors):
                handle.write(
                    f" {tangent[0]:12.6f} {tangent[1]:12.6f} {tangent[2]:12.6f}"
                    f" {chain_force[0]:16.6f} {chain_force[1]:12.6f} {chain_force[2]:12.6f}\n"
                )
            handle.write(" -------------------------------------------------------------------------------\n\n")
            handle.write(" CHAIN + TOTAL  (eV/Angst)\n")
            handle.write(" ----------------------------------------------\n")
            handle.write(
                f" {chain_plus_total[0]:12.5f} {chain_plus_total[1]:12.5f} {chain_plus_total[2]:12.5f}\n"
            )
            handle.write(" ----------------------------------------------\n\n")


def _append_oszicar_compat_step(
    recorder: _VaspCompatRecorder,
    step_index: int,
    *,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float,
    thermostat_potential: float,
    thermostat_kinetic: float,
    temperature: float,
    forces: np.ndarray | None = None,
) -> None:
    """Append one ionic step to ``OSZICAR``."""

    delta = 0.0 if recorder.previous_energy is None else potential_energy - recorder.previous_energy
    recorder.previous_energy = potential_energy

    with open("OSZICAR", "a", encoding="utf-8") as handle:
        if recorder.pseudo_scf.enabled:
            if not recorder.oszicar_scf_header_written:
                handle.write("       N       E                     dE             d eps       ncg     rms          rms(c)\n")
                recorder.oszicar_scf_header_written = True
            if forces is not None and forces.size:
                force_rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=1))))
                ncg = max(1, int(12 * len(forces)))
            else:
                force_rms = 0.0
                ncg = 1
            handle.write(
                f"DAV: {1:3d} {potential_energy:21.12E} "
                f"{_format_oszicar_residual(0.0):>14s} {_format_oszicar_residual(0.0):>14s} "
                f"{ncg:7d} {_format_oszicar_residual(force_rms):>12s}\n"
            )

        if recorder.ibrion == 0:
            handle.write(
                f"{step_index:7d} T={temperature:8.1f} "
                f"E= {_format_oszicar_energy(total_energy)} "
                f"F= {_format_oszicar_energy(potential_energy)} "
                f"E0= {_format_oszicar_energy(potential_energy)}  "
                f"EK= {_format_oszicar_energy(kinetic_energy)} "
                f"SP= {_format_oszicar_energy(thermostat_potential)} "
                f"SK= {_format_oszicar_energy(thermostat_kinetic)}\n"
            )
        else:
            handle.write(
                f"{step_index:4d} F= {_format_oszicar_energy(potential_energy)} "
                f"E0= {_format_oszicar_energy(potential_energy)}  "
                f"d E = {_format_oszicar_energy(delta)}\n"
            )


def _record_vasp_compat_step(
    recorder: _VaspCompatRecorder,
    atoms,
    *,
    step_index: int,
    potential_energy: float,
    total_energy: float,
    kinetic_energy: float = 0.0,
    thermostat_potential: float = 0.0,
    thermostat_kinetic: float = 0.0,
    temperature: float = 0.0,
    sc_time: float = 0.0,
    neb_chain: _NebChainApproximation | None = None,
) -> None:
    """Capture step data and append compatibility records."""

    forces = _safe_get_forces(atoms)
    stress_matrix = _safe_get_stress_matrix(atoms, mode=recorder.stress_mode)
    if recorder.neb_mode and neb_chain is None:
        neb_chain = _estimate_neb_chain_approximation(
            positions=np.asarray(atoms.get_positions(), dtype=float),
            forces=forces,
            prev_positions=recorder.neb_prev_positions,
            next_positions=recorder.neb_next_positions,
        )
    _append_outcar_compat_step(
        step_index,
        atoms,
        forces,
        stress_matrix,
        recorder.pseudo_scf,
        potential_energy,
        total_energy,
        kinetic_energy,
        thermostat_potential,
        thermostat_kinetic,
        recorder.neb_mode,
        neb_chain=neb_chain,
    )
    _append_oszicar_compat_step(
        recorder,
        step_index,
        potential_energy=potential_energy,
        total_energy=total_energy,
        kinetic_energy=kinetic_energy,
        thermostat_potential=thermostat_potential,
        thermostat_kinetic=thermostat_kinetic,
        temperature=temperature,
        forces=forces,
    )
    recorder.steps.append(
        _VasprunStep(
            cell=_matrix_to_nested_list(atoms.get_cell().array),
            scaled_positions=_matrix_to_nested_list(atoms.get_scaled_positions()),
            forces=forces.tolist(),
            stress=None if stress_matrix is None else stress_matrix.tolist(),
            potential_energy=float(potential_energy),
            total_energy=float(total_energy),
            kinetic_energy=float(kinetic_energy),
            thermostat_potential=float(thermostat_potential),
            thermostat_kinetic=float(thermostat_kinetic),
            temperature=float(temperature),
            sc_time=float(sc_time),
        )
    )


def _append_structure_xml(
    parent,
    *,
    cell: list[list[float]],
    scaled_positions: list[list[float]],
    name: str | None = None,
):
    """Append a minimal VASP-like ``structure`` node."""

    attrs = {"name": name} if name is not None else {}
    structure = ET.SubElement(parent, "structure", attrs)
    crystal = ET.SubElement(structure, "crystal")

    basis = ET.SubElement(crystal, "varray", {"name": "basis"})
    cell_array = np.asarray(cell, dtype=float)
    for vector in cell_array:
        ET.SubElement(basis, "v").text = (
            f"{vector[0]:16.8f} {vector[1]:16.8f} {vector[2]:16.8f}"
        )

    volume = float(abs(np.linalg.det(cell_array)))
    ET.SubElement(crystal, "i", {"name": "volume"}).text = f"{volume:16.8f}"

    rec_basis = ET.SubElement(crystal, "varray", {"name": "rec_basis"})
    if abs(np.linalg.det(cell_array)) > 1e-14:
        reciprocal = np.linalg.inv(cell_array).T
    else:
        reciprocal = np.zeros((3, 3), dtype=float)
    for vector in reciprocal:
        ET.SubElement(rec_basis, "v").text = (
            f"{vector[0]:16.8f} {vector[1]:16.8f} {vector[2]:16.8f}"
        )

    positions = ET.SubElement(structure, "varray", {"name": "positions"})
    for row in scaled_positions:
        ET.SubElement(positions, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"

    return structure


def _build_atominfo_xml(parent, symbols: List[str]) -> None:
    """Append a compact ``atominfo`` section."""

    atominfo = ET.SubElement(parent, "atominfo")
    ET.SubElement(atominfo, "atoms").text = str(len(symbols))

    counts: OrderedDict[str, int] = OrderedDict()
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    ET.SubElement(atominfo, "types").text = str(len(counts))

    atom_array = ET.SubElement(atominfo, "array", {"name": "atoms"})
    ET.SubElement(atom_array, "field", {"type": "string"}).text = "element"
    atom_set = ET.SubElement(atom_array, "set")
    for symbol in symbols:
        row = ET.SubElement(atom_set, "rc")
        ET.SubElement(row, "c").text = symbol

    type_array = ET.SubElement(atominfo, "array", {"name": "atomtypes"})
    ET.SubElement(type_array, "field", {"type": "int"}).text = "atomspertype"
    ET.SubElement(type_array, "field", {"type": "string"}).text = "element"
    ET.SubElement(type_array, "field", {"type": "float"}).text = "mass"
    ET.SubElement(type_array, "field", {"type": "float"}).text = "valence"
    ET.SubElement(type_array, "field", {"type": "string"}).text = "pseudopotential"
    type_set = ET.SubElement(type_array, "set")
    for symbol, count in counts.items():
        row = ET.SubElement(type_set, "rc")
        ET.SubElement(row, "c").text = str(count)
        ET.SubElement(row, "c").text = symbol
        ET.SubElement(row, "c").text = f"{1.0:8.4f}"
        ET.SubElement(row, "c").text = f"{0.0:8.4f}"
        ET.SubElement(row, "c").text = f"PAW_PBE {symbol}"


def _append_pseudo_scf_xml_step(parent, step: _VasprunStep) -> None:
    """Append one minimal ``scstep`` block for VASP XML reader compatibility."""

    scstep = ET.SubElement(parent, "scstep")
    ET.SubElement(scstep, "time", {"name": "dav"}).text = f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
    ET.SubElement(scstep, "time", {"name": "total"}).text = (
        f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
    )
    energy = ET.SubElement(scstep, "energy")
    ET.SubElement(energy, "i", {"name": "e_fr_energy"}).text = f"{step.potential_energy:16.8f}"
    ET.SubElement(energy, "i", {"name": "e_wo_entrp"}).text = f"{step.potential_energy:16.8f}"
    ET.SubElement(energy, "i", {"name": "e_0_energy"}).text = f"{step.potential_energy:16.8f}"


def _write_vasprun_xml(recorder: _VaspCompatRecorder, final_atoms) -> None:
    """Write a minimal ``vasprun.xml`` with ionic-step data."""

    root = ET.Element("modeling")
    generator = ET.SubElement(root, "generator")
    ET.SubElement(generator, "i", {"name": "program", "type": "string"}).text = "VPMDK"
    ET.SubElement(generator, "i", {"name": "version", "type": "string"}).text = "0"

    incar = ET.SubElement(root, "incar")
    ET.SubElement(incar, "i", {"name": "IBRION", "type": "int"}).text = str(recorder.ibrion)
    if recorder.isif is not None:
        ET.SubElement(incar, "i", {"name": "ISIF", "type": "int"}).text = str(recorder.isif)
    ET.SubElement(incar, "i", {"name": "NSW", "type": "int"}).text = str(len(recorder.steps))
    if recorder.pseudo_scf.enabled:
        ET.SubElement(incar, "i", {"name": "NELM", "type": "int"}).text = str(recorder.pseudo_scf.nelm)
        ET.SubElement(incar, "i", {"name": "NELMIN", "type": "int"}).text = str(recorder.pseudo_scf.nelmin)
        ET.SubElement(incar, "i", {"name": "NELMDL", "type": "int"}).text = str(recorder.pseudo_scf.nelmdl)
        ET.SubElement(incar, "i", {"name": "EDIFF", "type": "float"}).text = (
            f"{recorder.pseudo_scf.ediff:.8E}"
        )
    if recorder.potim is not None:
        ET.SubElement(incar, "i", {"name": "POTIM", "type": "float"}).text = f"{recorder.potim:.6f}"
    if recorder.mdalgo is not None:
        ET.SubElement(incar, "i", {"name": "MDALGO", "type": "int"}).text = str(recorder.mdalgo)

    _append_structure_xml(
        root,
        name="primitive_cell",
        cell=recorder.initial_cell,
        scaled_positions=recorder.initial_scaled_positions,
    )
    primitive_index = ET.SubElement(root, "varray", {"name": "primitive_index"})
    for index in range(1, len(recorder.symbols) + 1):
        ET.SubElement(primitive_index, "v").text = f"{index:9d} "

    _append_kpoints_xml(root)

    parameters = ET.SubElement(root, "parameters")
    electronic = ET.SubElement(parameters, "separator", {"name": "electronic"})
    ET.SubElement(electronic, "i", {"name": "NELM", "type": "int"}).text = str(
        recorder.pseudo_scf.nelm
    )
    if recorder.pseudo_scf.enabled:
        ET.SubElement(electronic, "i", {"name": "NELMDL", "type": "int"}).text = str(
            recorder.pseudo_scf.nelmdl
        )
        ET.SubElement(electronic, "i", {"name": "NELMIN", "type": "int"}).text = str(
            recorder.pseudo_scf.nelmin
        )
        ET.SubElement(electronic, "i", {"name": "EDIFF", "type": "float"}).text = (
            f"{recorder.pseudo_scf.ediff:.8E}"
        )
        ET.SubElement(electronic, "i", {"name": "NBANDS", "type": "int"}).text = str(
            max(1, 4 * len(recorder.symbols))
        )
    ionic = ET.SubElement(parameters, "separator", {"name": "ionic"})
    ET.SubElement(ionic, "i", {"name": "IBRION", "type": "int"}).text = str(recorder.ibrion)
    if recorder.isif is not None:
        ET.SubElement(ionic, "i", {"name": "ISIF", "type": "int"}).text = str(recorder.isif)
    ET.SubElement(ionic, "i", {"name": "NSW", "type": "int"}).text = str(len(recorder.steps))
    if recorder.potim is not None:
        ET.SubElement(ionic, "i", {"name": "POTIM", "type": "float"}).text = f"{recorder.potim:.6f}"

    _build_atominfo_xml(root, recorder.symbols)
    _append_structure_xml(
        root,
        name="initialpos",
        cell=recorder.initial_cell,
        scaled_positions=recorder.initial_scaled_positions,
    )

    for step in recorder.steps:
        calculation = ET.SubElement(root, "calculation")
        if recorder.pseudo_scf.enabled:
            _append_pseudo_scf_xml_step(calculation, step)
        _append_structure_xml(
            calculation,
            cell=step.cell,
            scaled_positions=step.scaled_positions,
        )

        forces = ET.SubElement(calculation, "varray", {"name": "forces"})
        for row in step.forces:
            ET.SubElement(forces, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"
        if step.stress is not None:
            stress = ET.SubElement(calculation, "varray", {"name": "stress"})
            for row in step.stress:
                ET.SubElement(stress, "v").text = f"{row[0]:16.8f} {row[1]:16.8f} {row[2]:16.8f}"

        energy = ET.SubElement(calculation, "energy")
        ET.SubElement(energy, "i", {"name": "e_fr_energy"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "e_wo_entrp"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "e_0_energy"}).text = f"{step.potential_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "kinetic"}).text = f"{step.kinetic_energy:16.8f}"
        ET.SubElement(energy, "i", {"name": "nosepot"}).text = f"{step.thermostat_potential:16.8f}"
        ET.SubElement(energy, "i", {"name": "nosekinetic"}).text = f"{step.thermostat_kinetic:16.8f}"
        ET.SubElement(energy, "i", {"name": "total"}).text = f"{step.total_energy:16.8f}"
        if recorder.pseudo_scf.enabled:
            ET.SubElement(calculation, "time", {"name": "totalsc"}).text = (
                f"{step.sc_time:8.2f} {step.sc_time:8.2f}"
            )

    _append_structure_xml(
        root,
        name="finalpos",
        cell=_matrix_to_nested_list(final_atoms.get_cell().array),
        scaled_positions=_matrix_to_nested_list(final_atoms.get_scaled_positions()),
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ")
    tree.write("vasprun.xml", encoding="utf-8", xml_declaration=True)


def read_structure(poscar_path: str, potcar_path: str | None = None):
    """Read POSCAR and reconcile species with POTCAR if necessary."""
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    if potcar_path and os.path.exists(potcar_path):
        try:
            potcar = Potcar.from_file(potcar_path)
            potcar_symbols = getattr(potcar, "symbols", [])
        except Exception:
            potcar_symbols = []
        normalized_potcar_symbols = _normalize_species_labels(potcar_symbols)
        if normalized_potcar_symbols:
            # check consistency
            if poscar.site_symbols and len(poscar.site_symbols) == len(normalized_potcar_symbols):
                normalized_poscar_symbols = _normalize_species_labels(poscar.site_symbols)
                if normalized_poscar_symbols != normalized_potcar_symbols:
                    print(
                        "Warning: species in POSCAR and POTCAR differ. "
                        f"Using POTCAR order: {normalized_potcar_symbols}"
                    )
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
                elif list(poscar.site_symbols) != normalized_potcar_symbols:
                    poscar.site_symbols = normalized_potcar_symbols
                    structure = poscar.structure
            elif not poscar.site_symbols:
                poscar.site_symbols = normalized_potcar_symbols
                structure = poscar.structure
    else:
        if not poscar.site_symbols:
            print("Warning: POSCAR has no species names and no POTCAR provided.")
    return structure


def _coerce_int_tag(value: str, tag_name: str) -> int:
    """Parse integer BCAR tag values with a descriptive error message."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {tag_name} value: {value!r}") from None


def _coerce_bool_tag(value: str, tag_name: str) -> bool:
    """Parse boolean-like BCAR tags with descriptive errors."""

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {tag_name} value: {value!r}")


def _looks_like_filesystem_path(value: str, *, suffixes: Iterable[str] = ()) -> bool:
    """Return whether a string likely denotes a local filesystem path."""

    altsep = os.path.altsep
    if os.path.sep in value or (altsep is not None and altsep in value):
        return True
    lowered = value.lower()
    return any(lowered.endswith(suffix.lower()) for suffix in suffixes)


def _list_matlantis_calc_modes() -> str:
    """Return comma-separated list of available Matlantis calc modes."""

    if EstimatorCalcMode is None:
        return ""
    members = getattr(EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and members:
        return ", ".join(sorted(members))
    candidates = [name for name in dir(EstimatorCalcMode) if name.isupper()]
    if candidates:
        return ", ".join(sorted(candidates))
    return ""


def _resolve_matlantis_calc_mode(name):
    """Return ``EstimatorCalcMode`` or passthrough string for Matlantis calc mode."""

    if EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis EstimatorCalcMode not available. Install pfp-api-client."
        )

    if isinstance(name, EstimatorCalcMode):
        return name

    if name is None:
        raise ValueError("MATLANTIS_CALC_MODE must not be None")

    text = str(name)
    normalized = text.upper()

    candidate = getattr(EstimatorCalcMode, normalized, None)
    if candidate is not None:
        return candidate

    members = getattr(EstimatorCalcMode, "__members__", None)
    if isinstance(members, dict) and normalized in members:
        return members[normalized]

    try:
        return EstimatorCalcMode[normalized]  # type: ignore[index]
    except Exception:
        pass

    try:
        return EstimatorCalcMode(normalized)  # type: ignore[call-arg]
    except Exception:
        pass

    return text


def _build_matlantis_calculator(bcar_tags: Dict[str, str]):
    """Create the Matlantis ASE calculator configured from BCAR tags."""

    if MatlantisEstimator is None or MatlantisASECalculator is None or EstimatorCalcMode is None:
        raise RuntimeError(
            "Matlantis calculator not available. Install pfp-api-client and dependencies."
        )

    model_version = (
        bcar_tags.get("MATLANTIS_MODEL_VERSION")
        or bcar_tags.get("MODEL_VERSION")
        or bcar_tags.get("MODEL")
        or "v8.0.0"
    )
    priority_raw = bcar_tags.get("MATLANTIS_PRIORITY") or bcar_tags.get("PRIORITY")
    priority = 50 if priority_raw is None else _coerce_int_tag(priority_raw, "MATLANTIS_PRIORITY")

    calc_mode_value = bcar_tags.get("MATLANTIS_CALC_MODE") or bcar_tags.get("CALC_MODE")
    calc_mode = _resolve_matlantis_calc_mode(calc_mode_value or "PBE")

    estimator_kwargs: Dict[str, Any] = {
        "model_version": model_version,
        "priority": priority,
        "calc_mode": calc_mode,
    }

    return MatlantisASECalculator(MatlantisEstimator(**estimator_kwargs))


def _build_orb_calculator(bcar_tags: Dict[str, str]):
    """Create the ORB ASE calculator configured from BCAR tags."""

    if ORBCalculator is None or ORB_PRETRAINED_MODELS is None:
        raise RuntimeError("ORB calculator not available. Install orb-models and dependencies.")

    model_name = bcar_tags.get("ORB_MODEL") or DEFAULT_ORB_MODEL
    model_factory = ORB_PRETRAINED_MODELS.get(model_name)
    if model_factory is None:
        supported = ", ".join(sorted(ORB_PRETRAINED_MODELS))
        raise ValueError(f"Unsupported ORB model '{model_name}'. Available: {supported}")

    device = bcar_tags.get("DEVICE")
    precision = bcar_tags.get("ORB_PRECISION") or "float32-high"
    compile_value = bcar_tags.get("ORB_COMPILE")
    compile_flag = None if compile_value is None else _coerce_bool_tag(compile_value, "ORB_COMPILE")
    weights_path = bcar_tags.get("MODEL")

    model = model_factory(
        weights_path=weights_path or None,
        device=device,
        precision=precision,
        compile=compile_flag,
        train=False,
    )

    return ORBCalculator(model, device=device)


def _load_matris_checkpoint_model(checkpoint_path: str, *, device: str | None):
    """Load a MatRIS model directly from a checkpoint file."""

    if MatRISModel is None:
        raise RuntimeError("MatRIS model loader not available. Install matris and dependencies.")

    import torch

    checkpoint_state = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model = MatRISModel.from_dict(checkpoint_state)
    return model.to(device or "cpu")


def _download_file_to_path(url: str, destination_path: str) -> None:
    """Download a file to a local path atomically."""

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    partial_path = f"{destination_path}.part"
    request = urllib.request.Request(url, headers={"User-Agent": "vpmdk"})
    try:
        with urllib.request.urlopen(request) as response, open(partial_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(partial_path, destination_path)
    except Exception:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise


def _ensure_matris_named_model_checkpoint(model_name: str) -> str | None:
    """Download a known MatRIS named model into the standard cache when needed."""

    download_info = _MATRIS_NAMED_MODEL_DOWNLOADS.get(model_name.lower())
    if download_info is None:
        return None

    checkpoint_filename, url = download_info
    cache_dir = os.path.expanduser("~/.cache/matris")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, checkpoint_filename)
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"MatRIS checkpoint not found, downloading to {checkpoint_path} ...")
        _download_file_to_path(url, checkpoint_path)
    return checkpoint_path


def _instantiate_matris_calculator(*, model, task: str, device: str | None):
    """Create a MatRIS ASE calculator from a preloaded model instance."""

    if MatRISCalculator is None:
        raise RuntimeError("MatRIS calculator not available. Install matris and dependencies.")

    calculator = MatRISCalculator.__new__(MatRISCalculator)
    Calculator.__init__(calculator)
    calculator.task = task
    calculator.device = device or "cpu"
    calculator.model = model
    calculator.stress_unit = units.GPa
    calculator.key = {"atoms_per_graph", "ref_energy", *task}
    return calculator


def _build_matris_calculator(bcar_tags: Dict[str, str]):
    """Create the MatRIS ASE calculator configured from BCAR tags."""

    if MatRISCalculator is None:
        raise RuntimeError("MatRIS calculator not available. Install matris and dependencies.")

    device = _resolve_device(bcar_tags.get("DEVICE"))
    task = (bcar_tags.get("MATRIS_TASK") or "efs").lower()
    model_value = bcar_tags.get("MODEL") or DEFAULT_MATRIS_MODEL
    graph_converter_algorithm = _resolve_graph_converter_algorithm(
        bcar_tags,
        backend_tag="MATRIS",
    )

    if os.path.exists(model_value):
        model = _load_matris_checkpoint_model(model_value, device=device)
        if graph_converter_algorithm is not None:
            model = _override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="MatRIS",
            )
        return _instantiate_matris_calculator(model=model, task=task, device=device)

    if _looks_like_filesystem_path(
        model_value,
        suffixes=(".ckpt", ".pt", ".pth", ".pth.tar", ".tar"),
    ):
        raise FileNotFoundError(f"MatRIS model not found: {model_value}")

    checkpoint_path = _ensure_matris_named_model_checkpoint(model_value)
    if checkpoint_path is not None:
        model = _load_matris_checkpoint_model(checkpoint_path, device=device)
        if graph_converter_algorithm is not None:
            model = _override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="MatRIS",
            )
        return _instantiate_matris_calculator(model=model, task=task, device=device)

    calculator = _call_with_optional_kwargs(
        MatRISCalculator,
        model=model_value,
        task=task,
        device=device,
        optional_kwargs={"graph_converter_algorithm": graph_converter_algorithm},
    )
    if graph_converter_algorithm is not None:
        model = getattr(calculator, "model", None)
        if model is not None:
            calculator.model = _override_model_graph_converter_algorithm(
                model,
                algorithm=graph_converter_algorithm,
                backend_name="MatRIS",
            )
    return calculator


def _normalize_eqnorm_key(value: str) -> str:
    """Return a separator-insensitive Eqnorm lookup key."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _match_eqnorm_variant(value: str | None) -> str | None:
    """Return a canonical Eqnorm variant when a value matches one."""

    if not value:
        return None

    normalized = _normalize_eqnorm_key(value)
    for variant, aliases in _EQNORM_VARIANT_ALIASES.items():
        if normalized in {_normalize_eqnorm_key(alias) for alias in aliases}:
            return variant
    return None


def _normalize_eqnorm_variant(value: str) -> str:
    """Validate and normalize an Eqnorm model variant."""

    variant = _match_eqnorm_variant(value)
    if variant is not None:
        return variant

    supported = ", ".join(sorted(_EQNORM_VARIANT_ALIASES))
    raise ValueError(f"Invalid EQNORM_VARIANT value: {value!r}. Available: {supported}")


def _resolve_eqnorm_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return Eqnorm named-model metadata for a model key or alias."""

    normalized = _normalize_eqnorm_key(model_name)
    for spec in _EQNORM_NAMED_MODELS.values():
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

    article_api_url = spec.get("article_api_url")
    expected_filename = spec["checkpoint_filename"]
    if article_api_url:
        request = urllib.request.Request(article_api_url, headers={"User-Agent": "vpmdk"})
        try:
            with urllib.request.urlopen(request) as response:
                payload = json.load(response)
            for file_info in payload.get("files", []):
                if file_info.get("name") == expected_filename and file_info.get("download_url"):
                    return str(file_info["download_url"])
        except Exception:
            pass

    return spec["checkpoint_url"]


def _ensure_eqnorm_named_model_checkpoint(model_name: str) -> tuple[Dict[str, Any], str]:
    """Download a known Eqnorm named model into the standard cache when needed."""

    spec = _resolve_eqnorm_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in _EQNORM_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported Eqnorm model '{model_name}'. Available: {supported}")

    cache_dir = os.path.expanduser("~/.cache/eqnorm")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, spec["checkpoint_filename"])
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"Eqnorm checkpoint not found, downloading to {checkpoint_path} ...")
        _download_file_to_path(_resolve_eqnorm_download_url(spec), checkpoint_path)
    return spec, checkpoint_path


def _resolve_eqnorm_variant(
    model_value: str,
    bcar_tags: Dict[str, str],
    *,
    named_variant: str | None = None,
) -> str:
    """Resolve the Eqnorm architecture variant from BCAR tags or a checkpoint path."""

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

    candidate = os.path.basename(model_value)
    while candidate:
        inferred = _match_eqnorm_variant(candidate)
        if inferred is not None:
            return inferred
        stem, ext = os.path.splitext(candidate)
        if not ext:
            break
        candidate = stem

    supported = ", ".join(sorted(_EQNORM_VARIANT_ALIASES))
    raise ValueError(
        "Eqnorm local checkpoints require EQNORM_VARIANT set to one of "
        f"{supported} when the variant cannot be inferred from the filename."
    )


def _stage_eqnorm_checkpoint(checkpoint_path: str, variant: str) -> str:
    """Expose a checkpoint at the cache path expected by the upstream calculator."""

    cache_dir = os.path.expanduser("~/.cache/eqnorm")
    os.makedirs(cache_dir, exist_ok=True)
    staged_path = os.path.join(cache_dir, f"{variant}.pt")
    source_path = os.path.abspath(checkpoint_path)

    if os.path.abspath(staged_path) == source_path:
        return staged_path

    try:
        if os.path.exists(staged_path) and os.path.samefile(staged_path, source_path):
            return staged_path
    except FileNotFoundError:
        pass

    if os.path.lexists(staged_path):
        os.remove(staged_path)

    try:
        os.symlink(source_path, staged_path)
    except Exception:
        shutil.copy2(source_path, staged_path)

    return staged_path


@contextmanager
def _temporarily_stage_eqnorm_local_checkpoint(checkpoint_path: str, variant: str):
    """Temporarily expose a local Eqnorm checkpoint without poisoning the named cache."""

    cache_dir = os.path.expanduser("~/.cache/eqnorm")
    os.makedirs(cache_dir, exist_ok=True)

    staged_path = os.path.join(cache_dir, f"{variant}.pt")
    source_path = os.path.abspath(checkpoint_path)
    if os.path.abspath(staged_path) == source_path:
        yield staged_path
        return

    try:
        if os.path.exists(staged_path) and os.path.samefile(staged_path, source_path):
            yield staged_path
            return
    except FileNotFoundError:
        pass

    backup_path = None
    if os.path.lexists(staged_path):
        backup_path = os.path.join(
            cache_dir,
            f".{variant}.vpmdk-backup-{time.time_ns()}.pt",
        )
        os.replace(staged_path, backup_path)

    staged_path = _stage_eqnorm_checkpoint(source_path, variant)
    try:
        yield staged_path
    finally:
        if os.path.lexists(staged_path):
            os.remove(staged_path)
        if backup_path is not None:
            os.replace(backup_path, staged_path)


def _ensure_eqnorm_torch_safe_globals() -> None:
    """Allowlist globals needed by e3nn constants on PyTorch 2.6+."""

    try:
        import torch.serialization

        torch.serialization.add_safe_globals([slice])
    except Exception:
        pass


def _build_eqnorm_calculator(bcar_tags: Dict[str, str]):
    """Create the Eqnorm ASE calculator configured from BCAR tags."""

    if EqnormCalculator is None:
        raise RuntimeError("Eqnorm calculator not available. Install eqnorm and dependencies.")

    model_value = bcar_tags.get("MODEL") or DEFAULT_EQNORM_MODEL
    device = _resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    compile_flag = False
    compile_value = bcar_tags.get("EQNORM_COMPILE")
    if compile_value is not None:
        compile_flag = _coerce_bool_tag(compile_value, "EQNORM_COMPILE")

    _ensure_eqnorm_torch_safe_globals()

    if os.path.exists(model_value):
        variant = _resolve_eqnorm_variant(model_value, bcar_tags)
        with _temporarily_stage_eqnorm_local_checkpoint(model_value, variant):
            return EqnormCalculator(
                model_name="eqnorm",
                model_variant=variant,
                device=device,
                compile=compile_flag,
            )
    if _looks_like_filesystem_path(model_value, suffixes=(".pt", ".pth", ".ckpt")):
        raise FileNotFoundError(f"Eqnorm model not found: {model_value}")

    spec, _ = _ensure_eqnorm_named_model_checkpoint(model_value)
    variant = _resolve_eqnorm_variant(
        model_value,
        bcar_tags,
        named_variant=spec["model_variant"],
    )
    return EqnormCalculator(
        model_name="eqnorm",
        model_variant=variant,
        device=device,
        compile=compile_flag,
    )


def _normalize_hienet_file_type(value: str | None) -> str:
    """Return the normalized HIENet file type."""

    if value is None:
        return "checkpoint"
    normalized = str(value).strip().lower()
    if normalized in {"checkpoint", "torchscript"}:
        return normalized
    raise ValueError(f"Invalid HIENET_FILE_TYPE value: {value!r}")


def _resolve_hienet_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return HIENet named-model metadata for a model key or alias."""

    normalized = model_name.strip().casefold()
    direct = _HIENET_NAMED_MODELS.get(normalized)
    if direct is not None:
        return direct

    for spec in _HIENET_NAMED_MODELS.values():
        aliases = spec.get("aliases", [])
        if any(normalized == alias.casefold() for alias in aliases):
            return spec
    return None


def _ensure_hienet_named_model_checkpoint(model_name: str) -> tuple[Dict[str, Any], str]:
    """Download a known HIENet named model into the standard cache when needed."""

    spec = _resolve_hienet_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in _HIENET_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported HIENet model '{model_name}'. Available: {supported}")

    cache_dir = os.path.expanduser("~/.cache/hienet")
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, spec["checkpoint_filename"])
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"HIENet checkpoint not found, downloading to {checkpoint_path} ...")
        _download_file_to_path(spec["checkpoint_url"], checkpoint_path)
    return spec, checkpoint_path


def _build_hienet_calculator(bcar_tags: Dict[str, str]):
    """Create the HIENet ASE calculator configured from BCAR tags."""

    if HIENetCalculator is None:
        raise RuntimeError("HIENet calculator not available. Install hienet and dependencies.")

    model_value = bcar_tags.get("MODEL") or DEFAULT_HIENET_MODEL
    device = _resolve_device(bcar_tags.get("DEVICE")) or "cpu"
    file_type = _normalize_hienet_file_type(bcar_tags.get("HIENET_FILE_TYPE"))

    model_path = model_value
    if os.path.exists(model_value):
        pass
    elif _looks_like_filesystem_path(
        model_value,
        suffixes=(".pth", ".pt", ".ckpt", ".jit", ".ts"),
    ):
        raise FileNotFoundError(f"HIENet model not found: {model_value}")
    else:
        if file_type != "checkpoint":
            raise ValueError(
                "HIENET_FILE_TYPE=torchscript requires MODEL pointing to a local TorchScript file."
            )
        _, model_path = _ensure_hienet_named_model_checkpoint(model_value)

    return HIENetCalculator(model=model_path, file_type=file_type, device=device)


def _normalize_nequix_backend(value: str | None) -> str:
    """Return the normalized Nequix backend name."""

    if value is None:
        return "jax"
    normalized = str(value).strip().lower()
    if normalized in {"jax", "torch"}:
        return normalized
    raise ValueError(f"Invalid NEQUIX_BACKEND value: {value!r}")


def _list_nequix_named_models() -> List[str]:
    """Return the named Nequix models exposed by the upstream calculator."""

    urls = getattr(NequixCalculator, "URLS", None)
    if isinstance(urls, dict):
        return sorted(str(name) for name in urls)
    return []


def _resolve_nequix_model_name(model_name: str) -> str:
    """Resolve a named Nequix model case-insensitively when metadata is available."""

    normalized = model_name.strip().casefold()
    supported = _list_nequix_named_models()
    for candidate in supported:
        if normalized == candidate.casefold():
            return candidate
    if supported:
        supported_text = ", ".join(supported)
        raise ValueError(f"Unsupported Nequix model '{model_name}'. Available: {supported_text}")
    return model_name


def _build_nequix_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create the Nequix ASE calculator configured from BCAR tags."""

    if NequixCalculator is None:
        raise RuntimeError("Nequix calculator not available. Install nequix and dependencies.")

    model_value = bcar_tags.get("MODEL") or DEFAULT_NEQUIX_MODEL
    backend = _normalize_nequix_backend(bcar_tags.get("NEQUIX_BACKEND"))

    use_kernel_tag = bcar_tags.get("NEQUIX_USE_KERNEL")
    if use_kernel_tag is None:
        use_kernel_tag = bcar_tags.get("NEQUIX_KERNEL")
    use_kernel = (
        _coerce_bool_tag(use_kernel_tag, "NEQUIX_USE_KERNEL")
        if use_kernel_tag is not None
        else False
    )

    use_compile_tag = bcar_tags.get("NEQUIX_USE_COMPILE")
    if use_compile_tag is None:
        use_compile_tag = bcar_tags.get("NEQUIX_COMPILE")
    use_compile = (
        _coerce_bool_tag(use_compile_tag, "NEQUIX_USE_COMPILE")
        if use_compile_tag is not None
        else False
    )

    capacity_multiplier = 1.1
    capacity_tag = bcar_tags.get("NEQUIX_CAPACITY_MULTIPLIER")
    if capacity_tag is not None:
        try:
            capacity_multiplier = float(capacity_tag)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid NEQUIX_CAPACITY_MULTIPLIER value: {capacity_tag!r}"
            ) from None

    kwargs: Dict[str, Any] = {
        "backend": backend,
        "use_kernel": use_kernel,
        "use_compile": use_compile,
        "capacity_multiplier": capacity_multiplier,
    }

    if os.path.exists(model_value):
        kwargs["model_path"] = model_value
        kwargs["model_name"] = os.path.splitext(os.path.basename(model_value))[0]
    elif _looks_like_filesystem_path(model_value, suffixes=(".nqx", ".pt", ".pth", ".ckpt")):
        raise FileNotFoundError(f"Nequix model not found: {model_value}")
    else:
        kwargs["model_name"] = _resolve_nequix_model_name(model_value)

    requested_device = bcar_tags.get("DEVICE")
    if backend == "torch":
        try:
            import torch

            nequix_module = importlib.import_module("nequix.calculator")
            nequix_data_module = importlib.import_module("nequix.data")

            torch_device = torch.device(
                _resolve_device(requested_device)
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model, config = nequix_module.from_pretrained(
                model_name=kwargs.get("model_name"),
                model_path=kwargs.get("model_path"),
                backend="torch",
                use_kernel=use_kernel,
            )

            calculator = NequixCalculator.__new__(NequixCalculator)
            Calculator.__init__(calculator)
            calculator.model = model.to(torch_device)
            calculator.config = config
            calculator.device = torch_device
            calculator.model.eval()
            calculator.compile_state = False if use_compile and torch_device.type == "cuda" else True
            calculator.atom_indices = nequix_data_module.atomic_numbers_to_indices(
                config["atomic_numbers"]
            )
            calculator.cutoff = config["cutoff"]
            calculator._capacity = None
            calculator._capacity_multiplier = capacity_multiplier
            calculator.backend = "torch"
            return calculator
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize Nequix torch backend for DEVICE={requested_device!r}."
            ) from exc

    return NequixCalculator(**kwargs)


def _normalize_alphanet_precision(value: str | None) -> str:
    """Return AlphaNet precision in the calculator's expected form."""

    if value is None:
        return "32"
    normalized = str(value).strip().lower()
    if normalized in {"32", "float32", "fp32"}:
        return "32"
    if normalized in {"64", "float64", "fp64"}:
        return "64"
    raise ValueError(f"Invalid ALPHANET_PRECISION value: {value!r}")


def _resolve_alphanet_named_model_spec(model_name: str) -> Dict[str, Any] | None:
    """Return AlphaNet named-model metadata for a model key or alias."""

    normalized = model_name.strip().casefold()
    direct = _ALPHANET_NAMED_MODELS.get(normalized)
    if direct is not None:
        return direct

    for spec in _ALPHANET_NAMED_MODELS.values():
        aliases = [spec["display_name"], *spec.get("aliases", [])]
        if normalized in {alias.casefold() for alias in aliases}:
            return spec
    return None


def _ensure_alphanet_named_model_files(model_name: str) -> tuple[str, str]:
    """Download a known AlphaNet named model and config when needed."""

    spec = _resolve_alphanet_named_model_spec(model_name)
    if spec is None:
        supported = ", ".join(
            sorted(named_spec["display_name"] for named_spec in _ALPHANET_NAMED_MODELS.values())
        )
        raise ValueError(f"Unsupported AlphaNet model '{model_name}'. Available: {supported}")

    cache_dir = os.path.join(
        os.path.expanduser("~/.cache/alphanet"),
        spec["display_name"].replace("/", "_"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    checkpoint_path = os.path.join(cache_dir, spec["checkpoint_filename"])
    config_path = os.path.join(cache_dir, spec["config_filename"])

    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        print(f"AlphaNet config not found, downloading to {config_path} ...")
        _download_file_to_path(spec["config_url"], config_path)

    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        print(f"AlphaNet checkpoint not found, downloading to {checkpoint_path} ...")
        _download_file_to_path(spec["checkpoint_url"], checkpoint_path)

    return checkpoint_path, config_path


def _resolve_alphanet_config_path(
    model_path: str,
    bcar_tags: Dict[str, str],
    *,
    default_config_path: str | None = None,
) -> str:
    """Resolve AlphaNet config JSON from BCAR or neighboring files."""

    config_path = bcar_tags.get("ALPHANET_CONFIG") or default_config_path
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AlphaNet config not found: {config_path}")
        return config_path

    parent_dir = os.path.dirname(model_path) or "."
    json_candidates = sorted(
        os.path.join(parent_dir, name)
        for name in os.listdir(parent_dir)
        if name.lower().endswith(".json")
    )
    if len(json_candidates) == 1:
        return json_candidates[0]

    raise ValueError(
        "AlphaNet requires ALPHANET_CONFIG pointing to a JSON config when it cannot "
        "be inferred from the checkpoint directory."
    )


def _load_alphanet_config(
    config_path: str,
    *,
    precision: str,
    use_pbc: bool,
    compute_stress: bool,
):
    """Load and normalize AlphaNet config for ASE inference."""

    if AlphaNetAllConfig is None:
        raise RuntimeError("AlphaNet config loader not available. Install AlphaNet and dependencies.")

    config = AlphaNetAllConfig.from_json(config_path)
    model_config = getattr(config, "model", config)
    model_config.compute_forces = True
    model_config.compute_stress = compute_stress
    model_config.use_pbc = use_pbc
    model_config.dtype = precision
    return config


def _build_alphanet_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Create the AlphaNet ASE calculator configured from BCAR tags."""

    if AlphaNetCalculator is None:
        raise RuntimeError("AlphaNet calculator not available. Install AlphaNet and dependencies.")

    model_value = bcar_tags.get("MODEL") or DEFAULT_ALPHANET_MODEL
    precision = _normalize_alphanet_precision(
        bcar_tags.get("ALPHANET_PRECISION") or bcar_tags.get("ALPHANET_DTYPE")
    )
    device = _resolve_device(bcar_tags.get("DEVICE")) or "cpu"

    config_path = None
    checkpoint_path = model_value

    if os.path.exists(model_value):
        config_path = _resolve_alphanet_config_path(model_value, bcar_tags)
    elif _looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
        raise FileNotFoundError(f"AlphaNet model not found: {model_value}")
    else:
        checkpoint_path, config_path = _ensure_alphanet_named_model_files(model_value)
        config_path = _resolve_alphanet_config_path(
            checkpoint_path,
            bcar_tags,
            default_config_path=config_path,
        )

    use_pbc = True if structure is None else getattr(structure, "lattice", None) is not None
    config = _load_alphanet_config(
        config_path,
        precision=precision,
        use_pbc=use_pbc,
        compute_stress=use_pbc,
    )

    return AlphaNetCalculator(
        ckpt_path=checkpoint_path,
        config=config,
        device=device,
        precision=precision,
    )


def _build_upet_calculator(bcar_tags: Dict[str, str]):
    """Create the UPET ASE calculator configured from BCAR tags."""

    if UPETCalculator is None:
        raise RuntimeError("UPET calculator not available. Install upet and dependencies.")

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "UPET requires MODEL set to a checkpoint path or a named model such as pet-oam-xl."
        )

    device = _resolve_device(bcar_tags.get("DEVICE"))
    kwargs: Dict[str, Any] = {"device": device}

    version = bcar_tags.get("UPET_VERSION")
    if version:
        kwargs["version"] = version

    non_conservative_value = bcar_tags.get("UPET_NON_CONSERVATIVE")
    if non_conservative_value is not None:
        kwargs["non_conservative"] = _coerce_bool_tag(
            non_conservative_value, "UPET_NON_CONSERVATIVE"
        )

    if os.path.exists(model_value):
        return UPETCalculator(checkpoint_path=model_value, **kwargs)

    if _looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
        raise FileNotFoundError(f"UPET model not found: {model_value}")

    return UPETCalculator(model=model_value, **kwargs)


def _callable_supports_parameter(callable_obj: object, parameter_name: str) -> bool:
    """Return whether a callable exposes a named parameter."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


def _build_tace_calculator(bcar_tags: Dict[str, str]):
    """Create the TACE ASE calculator configured from BCAR tags."""

    if TACEAseCalc is None:
        raise RuntimeError("TACE calculator not available. Install TACE and dependencies.")

    model_value = bcar_tags.get("MODEL")
    if not model_value:
        raise ValueError(
            "TACE requires MODEL set to a checkpoint path or a named model such as TACE-v1-OMat24-M."
        )

    model_path = model_value
    if not os.path.exists(model_value):
        if _looks_like_filesystem_path(model_value, suffixes=(".ckpt", ".pt", ".pth")):
            raise FileNotFoundError(f"TACE model not found: {model_value}")

        if tace_foundations is None:
            raise RuntimeError(
                "TACE named-model registry is not available. Install TACE with foundation-model "
                "support or provide MODEL as a local checkpoint path."
            )
        try:
            model_path = os.fspath(tace_foundations[model_value])
        except KeyError as exc:
            supported = (
                ", ".join(tace_foundations.list_models())
                if hasattr(tace_foundations, "list_models")
                else ""
            )
            if supported:
                raise ValueError(
                    f"Unsupported TACE model '{model_value}'. Available: {supported}"
                ) from exc
            raise ValueError(f"Unsupported TACE model '{model_value}'.") from exc

    kwargs: Dict[str, Any] = {
        "model": model_path,
        "device": _resolve_device(bcar_tags.get("DEVICE")),
    }

    dtype = bcar_tags.get("TACE_DTYPE")
    if dtype:
        kwargs["dtype"] = dtype

    spin_on_value = bcar_tags.get("TACE_SPIN_ON")
    if spin_on_value is not None:
        kwargs["spin_on"] = _coerce_bool_tag(spin_on_value, "TACE_SPIN_ON")

    neighborlist_backend = bcar_tags.get("TACE_NEIGHBORLIST_BACKEND")
    if neighborlist_backend:
        kwargs["neighborlist_backend"] = neighborlist_backend

    level_tag = None
    if "TACE_FIDELITY_IDX" in bcar_tags:
        level_tag = "TACE_FIDELITY_IDX"
    elif "TACE_LEVEL" in bcar_tags:
        level_tag = "TACE_LEVEL"

    if level_tag is not None:
        level_value = _coerce_int_tag(bcar_tags[level_tag], level_tag)
        if _callable_supports_parameter(TACEAseCalc, "fidelity_idx"):
            kwargs["fidelity_idx"] = level_value
        elif _callable_supports_parameter(TACEAseCalc, "level"):
            kwargs["level"] = level_value

    return TACEAseCalc(**kwargs)


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


def _get_fairchem_v1_calculator_cls():
    """Return FAIRChem v1 calculator class if installed."""

    global FAIRChemV1Calculator

    if FAIRChemV1Calculator is not None:
        return FAIRChemV1Calculator

    for module_name in _FAIRCHEM_V1_IMPORT_PATHS:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception:  # pragma: no cover - importlib edge case
            continue

        if spec is None:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional dependency
            continue

        candidate = getattr(module, "OCPCalculator", None)
        if candidate is not None:
            FAIRChemV1Calculator = candidate
            return candidate

    return None


def _get_fairchem_v1_predictor_cls():
    """Return FAIRChem v1 predictor class if installed."""

    global FAIRChemV1Predictor

    if FAIRChemV1Predictor is not None:
        return FAIRChemV1Predictor

    for module_name in _FAIRCHEM_V1_PREDICTOR_IMPORT_PATHS + _FAIRCHEM_V1_IMPORT_PATHS:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception:  # pragma: no cover - importlib edge case
            continue

        if spec is None:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional dependency
            continue

        for class_name in _FAIRCHEM_V1_PREDICTOR_CLASS_NAMES:
            candidate = getattr(module, class_name, None)
            if candidate is not None:
                FAIRChemV1Predictor = candidate
                return candidate

    return None


def _build_fairchem_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem ASE calculator configured from BCAR tags."""

    if FAIRChemCalculator is None:
        raise RuntimeError("FAIRChemCalculator not available. Install fairchem and dependencies.")

    model_name = bcar_tags.get("MODEL") or DEFAULT_FAIRCHEM_MODEL
    task_name = bcar_tags.get("FAIRCHEM_TASK")
    inference_settings = bcar_tags.get("FAIRCHEM_INFERENCE_SETTINGS") or "default"
    device = bcar_tags.get("DEVICE")

    return FAIRChemCalculator.from_model_checkpoint(
        model_name,
        task_name=task_name,
        inference_settings=inference_settings,
        device=device,
    )


def _pick_fairchem_prediction_value(prediction, keys: Iterable[str]):
    if isinstance(prediction, dict):
        for key in keys:
            if key in prediction:
                return prediction[key]
    for key in keys:
        if hasattr(prediction, key):
            return getattr(prediction, key)
    return None


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _normalize_fairchem_prediction(prediction, atoms):
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            prediction = {}
        else:
            prediction = prediction[0]

    energy_value = _pick_fairchem_prediction_value(
        prediction, ("energy", "energies", "y_energy", "y")
    )
    forces_value = _pick_fairchem_prediction_value(
        prediction, ("forces", "force", "y_force", "y_forces")
    )
    stress_value = _pick_fairchem_prediction_value(
        prediction, ("stress", "stresses", "virial")
    )

    energy = _as_numpy(energy_value)
    if energy is None:
        energy_float = 0.0
    else:
        energy_float = float(np.asarray(energy).reshape(-1)[0])

    forces = _as_numpy(forces_value)
    if forces is None:
        forces_array = np.zeros((len(atoms), 3))
    else:
        forces_array = np.asarray(forces)
        if forces_array.size == len(atoms) * 3:
            forces_array = forces_array.reshape((len(atoms), 3))

    stress = _as_numpy(stress_value)
    if stress is None:
        stress_array = np.zeros(6)
    else:
        stress_array = np.asarray(stress).reshape(-1)
        if stress_array.size == 9:
            stress_matrix = stress_array.reshape(3, 3)
            stress_array = np.array(
                [
                    stress_matrix[0, 0],
                    stress_matrix[1, 1],
                    stress_matrix[2, 2],
                    stress_matrix[1, 2],
                    stress_matrix[0, 2],
                    stress_matrix[0, 1],
                ]
            )
        elif stress_array.size != 6:
            stress_array = np.zeros(6)

    return energy_float, forces_array, stress_array


def _run_fairchem_v1_prediction(predictor, atoms):
    for method_name in ("predict_atoms", "predict", "__call__"):
        method = getattr(predictor, method_name, None)
        if not callable(method):
            continue
        try:
            return method(atoms)
        except TypeError:
            try:
                return method([atoms])
            except Exception:
                continue
    raise RuntimeError("FAIRChem v1 predictor does not expose a usable prediction method.")


class _FairChemV1PredictorCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, predictor):
        super().__init__()
        self._predictor = predictor

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            self.results = {"energy": 0.0, "forces": [], "stress": [0.0] * 6}
            return
        prediction = _run_fairchem_v1_prediction(self._predictor, atoms)
        energy, forces, stress = _normalize_fairchem_prediction(prediction, atoms)
        self.results = {"energy": energy, "forces": forces, "stress": stress}


def _build_fairchem_v1_predictor(bcar_tags: Dict[str, str]):
    """Create a FAIRChem v1 predictor-backed ASE calculator."""

    predictor_cls = _get_fairchem_v1_predictor_cls()
    if predictor_cls is None:
        raise RuntimeError(
            "FAIRChem v1 predictor not available. Install fairchem v1 (OCP) dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("FAIRChem v1 requires MODEL pointing to a checkpoint file.")

    config_path = bcar_tags.get("FAIRCHEM_CONFIG")
    device = bcar_tags.get("DEVICE")
    cpu_flag = device is not None and device.lower() == "cpu"

    kwargs: Dict[str, Any] = {"checkpoint_path": model_path, "cpu": cpu_flag}
    if config_path:
        kwargs["config_yml"] = config_path
    if device and not cpu_flag:
        kwargs["device"] = device

    predictor = predictor_cls(**kwargs)
    return _FairChemV1PredictorCalculator(predictor)


def _build_fairchem_v1_calculator(bcar_tags: Dict[str, str]):
    """Create the FAIRChem v1 OCPCalculator configured from BCAR tags."""

    predictor_tag = bcar_tags.get("FAIRCHEM_V1_PREDICTOR")
    if predictor_tag is not None and _coerce_bool_tag(
        predictor_tag, "FAIRCHEM_V1_PREDICTOR"
    ):
        return _build_fairchem_v1_predictor(bcar_tags)

    calculator_cls = _get_fairchem_v1_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "FAIRChem v1 calculator not available. Install fairchem v1 (OCP) dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("FAIRChem v1 requires MODEL pointing to a checkpoint file.")

    config_path = bcar_tags.get("FAIRCHEM_CONFIG")
    device = bcar_tags.get("DEVICE")
    cpu_flag = device is not None and device.lower() == "cpu"

    kwargs: Dict[str, Any] = {"checkpoint_path": model_path, "cpu": cpu_flag}
    if config_path:
        kwargs["config_yml"] = config_path

    calculator = calculator_cls(**kwargs)
    return _attach_fallback_calculator(calculator, bcar_tags)


def _build_grace_calculator(bcar_tags: Dict[str, str]):
    """Create a GRACE (TensorPotential) ASE calculator."""

    if TPCalculator is None:
        raise RuntimeError(
            "TPCalculator not available. Install grace-tensorpotential and dependencies."
        )

    grace_kwargs: Dict[str, Any] = {}

    pad_fraction = _parse_optional_float(
        bcar_tags.get("GRACE_PAD_NEIGHBORS_FRACTION"), key="GRACE_PAD_NEIGHBORS_FRACTION"
    )
    if pad_fraction is not None:
        grace_kwargs["pad_neighbors_fraction"] = pad_fraction

    pad_atoms_raw = bcar_tags.get("GRACE_PAD_ATOMS_NUMBER")
    if pad_atoms_raw is not None:
        grace_kwargs["pad_atoms_number"] = _coerce_int_tag(
            pad_atoms_raw, "GRACE_PAD_ATOMS_NUMBER"
        )

    recompilation_raw = bcar_tags.get("GRACE_MAX_RECOMPILATION")
    if recompilation_raw is not None:
        grace_kwargs["max_number_reduction_recompilation"] = _coerce_int_tag(
            recompilation_raw, "GRACE_MAX_RECOMPILATION"
        )

    min_dist = _parse_optional_float(bcar_tags.get("GRACE_MIN_DIST"), key="GRACE_MIN_DIST")
    if min_dist is not None:
        grace_kwargs["min_dist"] = min_dist

    float_dtype = bcar_tags.get("GRACE_FLOAT_DTYPE")
    if float_dtype:
        grace_kwargs["float_dtype"] = float_dtype

    model_value = bcar_tags.get("MODEL")
    if model_value and os.path.exists(model_value):
        return TPCalculator(model_value, **grace_kwargs)

    available_models = GRACE_MODEL_NAMES
    default_model = DEFAULT_GRACE_MODEL
    if available_models:
        default_model = default_model if default_model in available_models else available_models[0]

    if grace_fm is not None and available_models:
        selected = model_value or default_model
        if selected not in available_models:
            print(
                f"Warning: Unknown GRACE model '{selected}', using default {default_model} instead."
            )
            selected = default_model
        return grace_fm(selected, **grace_kwargs)

    if model_value:
        raise FileNotFoundError(f"GRACE model not found: {model_value}")

    raise RuntimeError(
        "GRACE calculator requires a MODEL path or available foundation models (grace_fm)."
    )


def _build_deepmd_calculator(bcar_tags: Dict[str, str], structure=None):
    """Create a DeePMD-kit calculator configured from BCAR tags."""

    if DeePMDCalculator is None:
        raise RuntimeError(
            "DeePMD-kit calculator not available. Install deepmd-kit and dependencies."
        )

    model_path = bcar_tags.get("MODEL")
    if not model_path:
        raise ValueError("DeePMD-kit requires MODEL pointing to a frozen model file.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DeePMD-kit model not found: {model_path}")

    type_map_value = bcar_tags.get("DEEPMD_TYPE_MAP")
    type_map: List[str] = []
    if type_map_value:
        type_map = [item for item in re.split(r"[\s,]+", type_map_value.strip()) if item]
    elif structure is not None:
        type_map = _infer_type_map(structure)

    kwargs: Dict[str, Any] = {}
    if type_map:
        kwargs["type_map"] = type_map

    head_value = bcar_tags.get("DEEPMD_HEAD")
    if head_value:
        kwargs["head"] = head_value

    return DeePMDCalculator(model=model_path, **kwargs)


_SIMPLE_CALCULATORS: Dict[str, tuple[str, str]] = {
    "SEVENNET": (
        "SevenNetCalculator",
        "SevenNetCalculator not available. Install sevennet.",
    ),
    "MATTERSIM": (
        "MatterSimCalculator",
        "MatterSimCalculator not available. Install mattersim and dependencies.",
    ),
}


_CALCULATOR_BUILDERS: Dict[str, str] = {
    "CHGNET": "_build_chgnet_calculator",
    "MATGL": "_build_m3gnet_calculator",
    "M3GNET": "_build_m3gnet_calculator",
    "MACE": "_build_mace_calculator",
    "EQNORM": "_build_eqnorm_calculator",
    "MATRIS": "_build_matris_calculator",
    "ALPHANET": "_build_alphanet_calculator",
    "HIENET": "_build_hienet_calculator",
    "NEQUIX": "_build_nequix_calculator",
    "ALLEGRO": "_build_allegro_calculator",
    "NEQUIP": "_build_nequip_calculator",
    "MATLANTIS": "_build_matlantis_calculator",
    "ORB": "_build_orb_calculator",
    "UPET": "_build_upet_calculator",
    "TACE": "_build_tace_calculator",
    "FAIRCHEM": "_build_fairchem_calculator",
    "FAIRCHEM_V2": "_build_fairchem_calculator",
    "ESEN": "_build_fairchem_calculator",
    "FAIRCHEM_V1": "_build_fairchem_v1_calculator",
    "GRACE": "_build_grace_calculator",
    "DEEPMD": "_build_deepmd_calculator",
}


def _build_calculator_from_init_factory(calculator, bcar_tags: Dict[str, str]):
    init = getattr(calculator.__class__, "__init__", None)
    closure = getattr(init, "__closure__", None)
    if not closure:
        return None
    mlp = _resolve_mlp_tag(bcar_tags, default="")
    for cell in closure:
        factory = cell.cell_contents
        if not callable(factory):
            continue
        try:
            candidate = factory(mlp)
        except TypeError:
            try:
                candidate = factory()
            except Exception:
                continue
        if hasattr(candidate, "get_potential_energy"):
            return candidate
    return None


def _attach_fallback_calculator(calculator, bcar_tags: Dict[str, str]):
    if hasattr(calculator, "get_potential_energy"):
        return calculator
    fallback = getattr(calculator, "calculator", None)
    if fallback is None or not hasattr(fallback, "get_potential_energy"):
        fallback = _build_calculator_from_init_factory(calculator, bcar_tags)
    if fallback is None:
        raise RuntimeError(
            "FAIRChem v1 calculator wrapper does not provide an inner ASE calculator."
        )
    setattr(calculator, "calculator", fallback)
    return calculator


def get_calculator(bcar_tags: Dict[str, str], *, structure=None):
    """Return ASE calculator based on BCAR tags."""

    mlp = _resolve_mlp_tag(bcar_tags)
    if mlp in _SIMPLE_CALCULATORS:
        calculator_attr, message = _SIMPLE_CALCULATORS[mlp]
        calculator_cls = globals().get(calculator_attr)
        return _build_simple_model_calculator(calculator_cls, bcar_tags, message)

    builder_entry = _CALCULATOR_BUILDERS.get(mlp)
    if builder_entry is None:
        raise ValueError(f"Unsupported MLP type: {mlp}")
    if callable(builder_entry):
        builder = builder_entry
        builder_name = getattr(builder_entry, "__name__", "")
    else:
        builder_name = builder_entry
        builder = globals().get(builder_name)
    if builder is None:
        raise RuntimeError(f"Calculator builder not available: {builder_entry}")

    try:
        builder_signature = inspect.signature(builder)
    except (TypeError, ValueError):
        builder_signature = None

    accepts_structure = builder_name == "_build_deepmd_calculator"
    if builder_signature is not None:
        accepts_structure = accepts_structure or "structure" in builder_signature.parameters
        accepts_structure = accepts_structure or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in builder_signature.parameters.values()
        )

    if accepts_structure:
        return builder(bcar_tags, structure=structure)
    return builder(bcar_tags)


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


def _append_outcar_footer(recorder: _VaspCompatRecorder) -> None:
    """Append simplified VASP-like timing/memory footer to ``OUTCAR``."""

    elapsed = max(time.perf_counter() - recorder.started_at, 0.0)
    peak_memory_kb = 0.0
    minor_page_faults = 0
    major_page_faults = 0
    voluntary_context_switches = 0
    involuntary_context_switches = 0
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak_memory_kb = float(usage.ru_maxrss)
            if sys.platform.startswith("darwin"):
                peak_memory_kb /= 1024.0
            minor_page_faults = int(getattr(usage, "ru_minflt", 0))
            major_page_faults = int(getattr(usage, "ru_majflt", 0))
            voluntary_context_switches = int(getattr(usage, "ru_nvcsw", 0))
            involuntary_context_switches = int(getattr(usage, "ru_nivcsw", 0))
        except Exception:
            peak_memory_kb = 0.0
            minor_page_faults = 0
            major_page_faults = 0
            voluntary_context_switches = 0
            involuntary_context_switches = 0

    with open("OUTCAR", "a", encoding="utf-8") as handle:
        handle.write(" General timing and accounting informations for this job:\n")
        handle.write(" ========================================================\n")
        handle.write(
            f"   Total CPU time used (sec):{elapsed:16.3f}\n"
            f"   User time (sec):{elapsed:25.3f}\n"
            f"   System time (sec):{0.0:23.3f}\n"
            f"   Elapsed time (sec):{elapsed:22.3f}\n"
            f"   Maximum memory used (kb):{peak_memory_kb:15.1f}\n"
            f"   Average memory used (kb):{peak_memory_kb:15.1f}\n"
            f"   Number of ionic steps:{len(recorder.steps):21d}\n\n"
            f"   Minor page faults:{minor_page_faults:25d}\n"
            f"   Major page faults:{major_page_faults:25d}\n"
            f"   Voluntary context switches:{voluntary_context_switches:15d}\n"
            f"   Involuntary context switches:{involuntary_context_switches:13d}\n\n"
        )


def _extract_numeric_attribute(obj, names: Iterable[str]) -> float:
    """Return first numeric attribute or method result from ``names``."""

    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            try:
                result = value()
            except Exception:
                continue
            try:
                return float(result)
            except (TypeError, ValueError):
                continue
        else:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def run_single_point(
    atoms,
    calculator,
    *,
    isif: int | None = None,
    oszicar_pseudo_scf: bool = False,
    neb_mode: bool = False,
    neb_prev_positions: np.ndarray | None = None,
    neb_next_positions: np.ndarray | None = None,
):
    atoms.calc = _resolve_calculator(calculator)
    recorder = _initialize_vasp_compat_outputs(
        atoms,
        ibrion=-1,
        isif=isif,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_mode=neb_mode,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    energy = atoms.get_potential_energy()
    delta = 0.0
    kinetic_energy = 0.0
    temperature = 0.0
    try:
        kinetic_energy = float(atoms.get_kinetic_energy())
    except Exception:
        kinetic_energy = 0.0
    try:
        temperature = float(atoms.get_temperature())
    except Exception:
        temperature = 0.0
    _record_vasp_compat_step(
        recorder,
        atoms,
        step_index=1,
        potential_energy=energy,
        total_energy=energy + kinetic_energy,
        kinetic_energy=kinetic_energy,
        temperature=temperature,
    )
    _write_vasprun_xml(recorder, atoms)
    _append_outcar_footer(recorder)
    write("CONTCAR", atoms, direct=True)
    print(
        f"{1:4d} F= {_format_energy_value(energy)} "
        f"E0= {_format_energy_value(energy)}  d E ={_format_energy_value(delta)}"
    )
    return energy


def _resolve_calculator(calculator):
    if hasattr(calculator, "get_potential_energy"):
        return calculator
    inner_calculator = getattr(calculator, "calculator", None)
    if inner_calculator is not None and hasattr(inner_calculator, "get_potential_energy"):
        return inner_calculator
    return calculator


KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208


@dataclass(frozen=True)
class IncarSettings:
    """Container for the INCAR parameters that drive the simulation."""

    nsw: int = 0
    ibrion: int = -1
    ediffg: float = -0.02
    isif: int = 2
    stress_isif: int = 2
    pstress: float | None = None
    tebeg: float = 300.0
    teend: float = 300.0
    potim: float = 2.0
    mdalgo: int = 0
    smass: float | None = None
    thermostat_params: Dict[str, float] = field(default_factory=dict)

    @property
    def energy_tolerance(self) -> float | None:
        """Energy convergence threshold in eV when EDIFFG>0."""

        return self.ediffg if self.ediffg > 0 else None

    @property
    def force_limit(self) -> float:
        """Return ASE ``fmax`` argument derived from EDIFFG semantics."""

        if self.ediffg > 0:
            return -abs(self.ediffg)
        if self.ediffg < 0:
            return abs(self.ediffg)
        return 0.05


SUPPORTED_INCAR_TAGS = {
    "ISIF",
    "IBRION",
    "NSW",
    "EDIFFG",
    "PSTRESS",
    "TEBEG",
    "TEEND",
    "POTIM",
    "MDALGO",
    "SMASS",
    "ANDERSEN_PROB",
    "LANGEVIN_GAMMA",
    "CSVR_PERIOD",
    "NHC_NCHAINS",
    "MAGMOM",
    "IMAGES",
    "LCLIMB",
    "SPRING",
}

SUPPORTED_ISIF_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8}


_NUMERIC_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NEB_IMAGE_DIR_RE = re.compile(r"^\d+$")


def _load_incar(path: str):
    """Return ``Incar`` contents when available, falling back to ``{}``."""

    if os.path.exists(path):
        return Incar.from_file(path)
    return {}


def _warn_for_unsupported_incar_tags(incar, *, pseudo_scf_enabled: bool = False) -> None:
    """Emit warnings for INCAR options that are silently ignored."""

    supported_tags = SUPPORTED_INCAR_TAGS
    for key in getattr(incar, "keys", lambda: [])():
        if key in supported_tags:
            continue
        if pseudo_scf_enabled and key in _PSEUDO_SCF_INCAR_TAGS:
            print(
                f"Warning: INCAR tag {key} does not affect the run and is used only "
                "for pseudo-SCF compatibility output"
            )
            continue
        if key not in supported_tags:
            print(f"INCAR tag {key} is not supported and will be ignored")


def _is_truthy_flag(value) -> bool:
    """Return whether ``value`` expresses a truthy INCAR-style flag."""

    if value is None:
        return False
    token = str(value).strip().strip(".").upper()
    return token in {"T", "TRUE", "1", "YES", "Y"}


def _is_neb_like_incar(incar) -> bool:
    """Detect whether INCAR appears to describe a NEB-style calculation."""

    if not hasattr(incar, "get"):
        return False

    images_value = incar.get("IMAGES")
    if images_value is not None:
        match = _NUMERIC_RE.search(str(images_value))
        if match is not None:
            try:
                if int(float(match.group(0))) > 0:
                    return True
            except ValueError:
                pass

    if "SPRING" in getattr(incar, "keys", lambda: [])():
        return True

    if _is_truthy_flag(incar.get("LCLIMB")):
        return True

    return False


def _parse_neb_image_count(incar) -> int | None:
    """Return ``IMAGES`` value when parseable and non-negative."""

    if not hasattr(incar, "get"):
        return None
    raw_value = incar.get("IMAGES")
    if raw_value is None:
        return None
    parsed = _parse_optional_float(raw_value, key="IMAGES")
    if parsed is None:
        return None
    count = int(parsed)
    if count < 0:
        print(f"Warning: IMAGES={raw_value} is invalid; ignoring NEB image count hint.")
        return None
    return count


def _discover_neb_image_directories(workdir: str) -> List[str]:
    """Return numbered NEB image directories sorted by numeric index."""

    try:
        entries = os.listdir(workdir)
    except OSError:
        return []

    indexed_dirs: list[tuple[int, str]] = []
    for entry in entries:
        if _NEB_IMAGE_DIR_RE.fullmatch(entry) is None:
            continue
        path = os.path.join(workdir, entry)
        if os.path.isdir(path):
            indexed_dirs.append((int(entry), path))
    indexed_dirs.sort(key=lambda item: item[0])
    return [path for _, path in indexed_dirs]


def _resolve_neb_image_structure_path(image_dir: str, *, prefer_contcar: bool = False) -> str:
    """Return structure path for one NEB image (POSCAR/CONTCAR)."""

    poscar_path = os.path.join(image_dir, "POSCAR")
    contcar_path = os.path.join(image_dir, "CONTCAR")
    candidates = (
        (contcar_path, poscar_path) if prefer_contcar else (poscar_path, contcar_path)
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Neither POSCAR nor CONTCAR found in NEB image directory: {image_dir}"
    )


def _parse_vasprun_varray_rows(varray) -> np.ndarray:
    """Return numeric rows from a ``vasprun.xml`` ``varray`` element."""

    rows: list[list[float]] = []
    for vector in varray.findall("v"):
        parts = str(vector.text or "").split()
        if not parts:
            continue
        rows.append([float(value) for value in parts])
    return np.asarray(rows, dtype=float)


def _read_last_vasprun_step(path: str) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Return ``(energy, forces, stress)`` from last ``calculation`` in ``vasprun.xml``."""

    root = ET.parse(path).getroot()
    calculations = root.findall("calculation")
    if not calculations:
        raise ValueError("vasprun.xml has no <calculation> blocks")
    calculation = calculations[-1]

    energy_value: float | None = None
    energy = calculation.find("energy")
    if energy is not None:
        for name in ("e_wo_entrp", "e_fr_energy", "e_0_energy", "total"):
            node = energy.find(f"./i[@name='{name}']")
            if node is None or node.text is None:
                continue
            try:
                energy_value = float(node.text)
                break
            except ValueError:
                continue
    if energy_value is None:
        raise ValueError("Unable to parse energy from vasprun.xml")

    forces_varray = calculation.find("./varray[@name='forces']")
    forces = _parse_vasprun_varray_rows(forces_varray) if forces_varray is not None else None

    stress_varray = calculation.find("./varray[@name='stress']")
    stress = _parse_vasprun_varray_rows(stress_varray) if stress_varray is not None else None
    if stress is not None and stress.shape != (3, 3):
        stress = None

    return energy_value, forces, stress


def _collect_neb_image_results(
    image_dirs: list[str], *, potcar_path: str | None
) -> list[_NebImageResult]:
    """Collect final structures/energies/forces for each NEB image directory."""

    results: list[_NebImageResult] = []
    for image_dir in image_dirs:
        image_name = os.path.basename(image_dir)
        structure_path = _resolve_neb_image_structure_path(image_dir, prefer_contcar=True)
        structure = read_structure(structure_path, potcar_path)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.wrap()

        potential_energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=float)
        stress: np.ndarray | None = None
        vasprun_path = os.path.join(image_dir, "vasprun.xml")
        if os.path.exists(vasprun_path):
            try:
                potential_energy, parsed_forces, parsed_stress = _read_last_vasprun_step(vasprun_path)
                if parsed_forces is None or parsed_forces.shape != (len(atoms), 3):
                    raise ValueError(
                        f"Unexpected forces shape in {vasprun_path}: "
                        f"{None if parsed_forces is None else parsed_forces.shape}"
                    )
                forces = parsed_forces
                if parsed_stress is not None:
                    stress = parsed_stress
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse NEB image vasprun.xml for {image_name}: {vasprun_path}"
                ) from exc

        results.append(
            _NebImageResult(
                image_name=image_name,
                atoms=atoms,
                potential_energy=float(potential_energy),
                forces=forces,
                stress=stress,
            )
        )
    return results


def _write_neb_parent_aggregate_outputs(
    *,
    workdir: str,
    settings: IncarSettings,
    image_results: list[_NebImageResult],
    oszicar_pseudo_scf: bool = False,
) -> None:
    """Write parent-level NEB ``OUTCAR``/``OSZICAR``/``vasprun.xml`` summaries."""

    if not image_results:
        return

    first_atoms = image_results[0].atoms.copy()
    recorder = _initialize_vasp_compat_outputs(
        first_atoms,
        ibrion=settings.ibrion,
        isif=settings.stress_isif,
        potim=settings.potim,
        mdalgo=settings.mdalgo if settings.ibrion == 0 else None,
        neb_mode=True,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
    )
    image_positions = [np.asarray(image.atoms.get_positions(), dtype=float) for image in image_results]
    for image_index, image in enumerate(image_results):
        step_index = image_index + 1
        atoms_step = image.atoms.copy()
        prev_positions = image_positions[image_index - 1] if image_index > 0 else None
        next_positions = (
            image_positions[image_index + 1] if image_index + 1 < len(image_positions) else None
        )
        neb_chain = _estimate_neb_chain_approximation(
            positions=np.asarray(atoms_step.get_positions(), dtype=float),
            forces=np.asarray(image.forces, dtype=float),
            prev_positions=prev_positions,
            next_positions=next_positions,
        )
        calculator_kwargs: Dict[str, Any] = {
            "energy": image.potential_energy,
            "forces": image.forces,
        }
        if image.stress is not None:
            calculator_kwargs["stress"] = _full_to_voigt_stress(np.asarray(image.stress, dtype=float))
        atoms_step.calc = SinglePointCalculator(atoms_step, **calculator_kwargs)
        _record_vasp_compat_step(
            recorder,
            atoms_step,
            step_index=step_index,
            potential_energy=image.potential_energy,
            total_energy=image.potential_energy,
            sc_time=0.0,
            neb_chain=neb_chain,
        )

    final_atoms = image_results[-1].atoms.copy()
    _write_vasprun_xml(recorder, final_atoms)
    _append_outcar_footer(recorder)


def _parse_optional_float(value, *, key: str):
    """Attempt to convert ``value`` to ``float`` with warning on failure."""

    if value is None:
        return None
    candidate = value
    if isinstance(value, str):
        match = _NUMERIC_RE.search(value)
        if match is not None:
            candidate = match.group(0)
        else:
            candidate = value.strip()
    try:
        return float(candidate)
    except (TypeError, ValueError):
        print(f"Warning: Unable to parse {key}; ignoring value {value}")
        return None


def _normalize_isif(requested: int) -> int:
    """Map request to supported ISIF behaviour while preserving warnings."""

    if requested not in SUPPORTED_ISIF_VALUES:
        print(
            "Warning: ISIF="
            f"{requested} is not fully supported; defaulting to ISIF=2 behavior."
        )
        return 2
    if requested in (0, 1, 2):
        return 2
    return requested


def _extract_thermostat_parameters(incar) -> Dict[str, float]:
    """Collect thermostat keywords from ``incar`` with validation."""

    params: Dict[str, float] = {}
    keys = ("ANDERSEN_PROB", "LANGEVIN_GAMMA", "CSVR_PERIOD", "NHC_NCHAINS")
    for key in keys:
        if hasattr(incar, "__contains__") and key in incar:
            value = incar[key]
            if key == "NHC_NCHAINS":
                try:
                    coerced = int(float(value))
                except (TypeError, ValueError):
                    print(f"Warning: Unable to parse {key}; ignoring value {value}")
                    parsed = None
                else:
                    parsed = _parse_optional_float(coerced, key=key)
            else:
                parsed = _parse_optional_float(value, key=key)
            if parsed is not None:
                params[key] = float(parsed)
    return params


def _load_incar_settings(incar) -> IncarSettings:
    """Translate INCAR dictionary-like object into :class:`IncarSettings`."""

    if not hasattr(incar, "get"):
        return IncarSettings()

    nsw = int(float(incar.get("NSW", 0)))
    ibrion = int(float(incar.get("IBRION", -1)))
    ediffg = float(incar.get("EDIFFG", -0.02))
    pstress = None
    if "PSTRESS" in incar:
        pstress = _parse_optional_float(incar.get("PSTRESS", 0.0), key="PSTRESS")
    tebeg_default = 300.0
    tebeg_value = incar.get("TEBEG", tebeg_default)
    parsed_tebeg = _parse_optional_float(tebeg_value, key="TEBEG")
    tebeg = parsed_tebeg if parsed_tebeg is not None else tebeg_default

    teend_value = incar.get("TEEND", tebeg)
    parsed_teend = _parse_optional_float(teend_value, key="TEEND")
    teend = parsed_teend if parsed_teend is not None else tebeg
    potim = float(incar.get("POTIM", 2.0))
    smass = (
        _parse_optional_float(incar.get("SMASS"), key="SMASS")
        if "SMASS" in incar
        else None
    )
    mdalgo = int(float(incar.get("MDALGO", 0)))
    if mdalgo == 0 and smass is not None:
        if smass < 0:
            mdalgo = 3
        elif smass > 0:
            mdalgo = 2
    thermostat_params = _extract_thermostat_parameters(incar)
    default_isif = 0 if ibrion == 0 else 2
    requested_isif = int(float(incar.get("ISIF", default_isif)))
    normalized_isif = _normalize_isif(requested_isif)
    stress_isif = (
        requested_isif if requested_isif in SUPPORTED_ISIF_VALUES else normalized_isif
    )

    return IncarSettings(
        nsw=nsw,
        ibrion=ibrion,
        ediffg=ediffg,
        isif=normalized_isif,
        stress_isif=stress_isif,
        pstress=pstress,
        tebeg=tebeg,
        teend=teend,
        potim=potim,
        mdalgo=mdalgo,
        smass=smass,
        thermostat_params=thermostat_params,
    )


def _should_write_energy_csv(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests CSV output of ionic energies."""

    value = str(bcar_tags.get("WRITE_ENERGY_CSV", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_lammps_trajectory(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests LAMMPS-style trajectory output."""

    value = str(bcar_tags.get("WRITE_LAMMPS_TRAJ", "0")).lower()
    return value in {"1", "true", "yes", "on"}


def _should_write_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Return ``True`` when BCAR requests pseudo electronic-step compatibility output."""

    raw = bcar_tags.get("WRITE_PSEUDO_SCF", bcar_tags.get("WRITE_OSZICAR_PSEUDO_SCF", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _should_write_oszicar_pseudo_scf(bcar_tags: Dict[str, str]) -> bool:
    """Backward-compatible alias for :func:`_should_write_pseudo_scf`."""

    return _should_write_pseudo_scf(bcar_tags)


def _get_lammps_trajectory_interval(bcar_tags: Dict[str, str]) -> int:
    """Return the LAMMPS trajectory write interval requested in BCAR."""

    raw = bcar_tags.get("LAMMPS_TRAJ_INTERVAL", "1")
    interval = _coerce_int_tag(raw, "LAMMPS_TRAJ_INTERVAL")
    if interval <= 0:
        raise ValueError("LAMMPS_TRAJ_INTERVAL must be at least 1")
    return interval


class _EnergyConvergenceMonitor:
    """Track ionic step energies and test for convergence."""

    def __init__(self, atoms, tolerance: float):
        self._atoms = atoms
        self._tolerance = tolerance
        self._previous: float | None = None

    def update(self) -> bool:
        """Return True when the total energy change falls below the tolerance."""

        energy = self._atoms.get_potential_energy()
        if self._previous is None:
            self._previous = energy
            return False
        delta = abs(energy - self._previous)
        self._previous = energy
        return delta <= self._tolerance


def _make_relaxation_builder(
    isif: int,
    scalar_pressure: float | None,
    scalar_pressure_kwarg: float,
) -> tuple[Callable[[object], object], bool]:
    """Return a factory for the relaxation object and freeze requirement."""

    def build_identity(atoms):
        return atoms

    if isif == 3:
        if scalar_pressure is None:
            return UnitCellFilter, False

        def build_ucf(atoms):
            return UnitCellFilter(atoms, scalar_pressure=scalar_pressure)

        return build_ucf, False

    if isif == 4:

        def build_constant_volume(atoms):
            return UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume, False

    if isif == 5:

        def build_constant_volume_frozen(atoms):
            return UnitCellFilter(
                atoms,
                constant_volume=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_constant_volume_frozen, True

    if isif == 6:
        return StrainFilter, False

    if isif == 7:

        def build_hydrostatic_frozen(atoms):
            return UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic_frozen, True

    if isif == 8:

        def build_hydrostatic(atoms):
            return UnitCellFilter(
                atoms,
                mask=[1, 1, 1, 0, 0, 0],
                hydrostatic_strain=True,
                scalar_pressure=scalar_pressure_kwarg,
            )

        return build_hydrostatic, False

    return build_identity, False


@contextmanager
def _temporarily_freeze_atoms(atoms, freeze_required: bool):
    """Temporarily constrain ionic positions when required by ISIF."""

    if not freeze_required:
        yield
        return

    current_constraints = getattr(atoms, "constraints", None)
    if current_constraints is None:
        original_constraints = None
        base_constraints: list[object] = []
    else:
        try:
            base_constraints = list(current_constraints)
        except TypeError:
            base_constraints = [current_constraints]
        original_constraints = base_constraints

    frozen = FixAtoms(indices=list(range(len(atoms))))
    atoms.set_constraint(base_constraints + [frozen])
    try:
        yield
    finally:
        if original_constraints is None:
            atoms.set_constraint()
        else:
            atoms.set_constraint(original_constraints)


@contextmanager
def _working_directory(path: str):
    """Temporarily change the current working directory."""

    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def run_relaxation(
    atoms,
    calculator,
    steps: int,
    fmax: float,
    write_energy_csv: bool = False,
    isif: int = 2,
    pstress: float | None = None,
    energy_tolerance: float | None = None,
    ibrion: int = 2,
    stress_isif: int | None = None,
    neb_mode: bool = False,
    neb_prev_positions: np.ndarray | None = None,
    neb_next_positions: np.ndarray | None = None,
    oszicar_pseudo_scf: bool = False,
):
    atoms.calc = _resolve_calculator(calculator)
    recorder = _initialize_vasp_compat_outputs(
        atoms,
        ibrion=ibrion,
        isif=isif if stress_isif is None else stress_isif,
        neb_mode=neb_mode,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    energies: List[float] = []
    previous_energy: float | None = None
    step_counter = 0
    scalar_pressure = pstress * KBAR_TO_EV_PER_A3 if pstress is not None else None
    scalar_pressure_kwarg = scalar_pressure if scalar_pressure is not None else 0.0

    builder, freeze_required = _make_relaxation_builder(
        isif, scalar_pressure, scalar_pressure_kwarg
    )

    with _temporarily_freeze_atoms(atoms, freeze_required):
        relax_object = builder(atoms)
        dyn = BFGS(relax_object, logfile="OUTCAR")

        def _log_relaxation_energy() -> None:
            nonlocal previous_energy, step_counter
            target = getattr(relax_object, "atoms", atoms)
            energy = target.get_potential_energy()
            delta = 0.0 if previous_energy is None else energy - previous_energy
            previous_energy = energy
            step_counter += 1
            _record_vasp_compat_step(
                recorder,
                target,
                step_index=step_counter,
                potential_energy=energy,
                total_energy=energy,
            )
            print(
                f"{step_counter:4d} F= {_format_energy_value(energy)} "
                f"E0= {_format_energy_value(energy)}  d E ={_format_energy_value(delta)}"
            )

        if write_energy_csv:
            dyn.attach(lambda: energies.append(atoms.get_potential_energy()))
        dyn.attach(_log_relaxation_energy)
        if energy_tolerance is None:
            dyn.run(fmax=fmax, steps=steps)
        else:
            monitor = _EnergyConvergenceMonitor(atoms, energy_tolerance)
            dyn.fmax = fmax
            for force_converged in dyn.irun(steps=steps):
                energy_converged = monitor.update()
                if energy_converged or force_converged:
                    break

    target_atoms = getattr(relax_object, "atoms", atoms)
    target_atoms.wrap()
    if not recorder.steps:
        fallback_energy = target_atoms.get_potential_energy()
        _record_vasp_compat_step(
            recorder,
            target_atoms,
            step_index=1,
            potential_energy=fallback_energy,
            total_energy=fallback_energy,
        )
    _write_vasprun_xml(recorder, target_atoms)
    _append_outcar_footer(recorder)
    write("CONTCAR", target_atoms, direct=True)
    if write_energy_csv:
        with open("energy.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for energy in energies:
                writer.writerow([energy])
    return target_atoms.get_potential_energy()


def _rescale_velocities(atoms, target_temperature: float) -> None:
    """Scale velocities so that kinetic temperature approaches target."""

    if target_temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
        return

    ndof = getattr(atoms, "get_number_of_degrees_of_freedom", lambda: 0)()
    if ndof <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    kinetic_energy = atoms.get_kinetic_energy()
    if kinetic_energy <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    current_temperature = 2.0 * kinetic_energy / (ndof * units.kB)
    if current_temperature <= 0:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return

    scaling = (target_temperature / current_temperature) ** 0.5
    velocities = atoms.get_velocities()
    if velocities is None:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=target_temperature
        )
        return
    atoms.set_velocities(velocities * scaling)


def _estimate_tdamp(smass: float | None, timestep: float) -> float:
    """Return Nose-Hoover time constant (in fs)."""

    if smass is None or smass == 0:
        return max(100.0 * timestep, timestep)
    return abs(smass)


def _select_md_dynamics(
    atoms,
    mdalgo: int,
    timestep: float,
    initial_temperature: float,
    smass: float | None,
    thermostat_params: Dict[str, float],
):
    """Create ASE molecular dynamics driver and temperature updater."""

    timestep_ase = timestep * units.fs

    def default_update(temp: float) -> None:
        _rescale_velocities(atoms, temp)

    def make_update(dyn, *, allow_attribute_update: bool = False):
        def update(temp: float) -> None:
            try:
                dyn.set_temperature(temperature_K=temp)
            except TypeError:
                dyn.set_temperature(temp)
            except AttributeError:
                if not allow_attribute_update:
                    raise
                dyn.temp = temp * units.kB
                dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof
            _rescale_velocities(atoms, temp)

        return update

    if mdalgo == 1:
        if Andersen is None:
            raise RuntimeError(
                "Andersen thermostat requested but ase.md.andersen.Andersen "
                "is unavailable. Install the optional ASE thermostat "
                "dependencies or choose a supported MDALGO value."
            )
        andersen_prob = float(thermostat_params.get("ANDERSEN_PROB", 0.1))
        dyn = Andersen(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            andersen_prob=andersen_prob,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo in (2, 4) and NoseHooverChainNVT is not None:
        tdamp_fs = _estimate_tdamp(smass, timestep)
        if mdalgo == 2:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 1))
        else:
            chain_length = int(thermostat_params.get("NHC_NCHAINS", 3))
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_ase,
            temperature_K=initial_temperature,
            tdamp=tdamp_fs * units.fs,
            tchain=chain_length,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)
    if mdalgo in (2, 4) and NoseHooverChainNVT is None and mdalgo != 0:
        raise RuntimeError(
            "Nose-Hoover thermostat requested but ase.md.nose_hoover_chain.NoseHooverChainNVT "
            "is unavailable. Install the optional ASE thermostat dependencies or choose "
            "a supported MDALGO value."
        )

    if mdalgo == 3:
        if Langevin is None:
            raise RuntimeError(
                "Langevin thermostat requested but ase.md.langevin.Langevin "
                "is unavailable. Install the optional ASE thermostat dependencies or "
                "choose a supported MDALGO value."
            )
        gamma = thermostat_params.get("LANGEVIN_GAMMA")
        if gamma is None and smass is not None and smass < 0:
            gamma = abs(smass)
        if gamma is None:
            gamma = 1.0
        friction = (float(gamma) / 1000.0) / units.fs
        dyn = Langevin(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            friction=friction,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn)

    if mdalgo == 5:
        if Bussi is None:
            raise RuntimeError(
                "CSVR thermostat requested but ase.md.bussi.Bussi is unavailable. "
                "Install the optional ASE thermostat dependencies or choose a supported "
                "MDALGO value."
            )
        taut = thermostat_params.get("CSVR_PERIOD")
        if taut is None:
            taut = max(100.0 * timestep, timestep)
        dyn = Bussi(
            atoms,
            timestep_ase,
            temperature_K=initial_temperature,
            taut=float(taut) * units.fs,
            logfile="OUTCAR",
        )

        return dyn, make_update(dyn, allow_attribute_update=True)

    dyn = VelocityVerlet(atoms, timestep_ase, logfile="OUTCAR")
    return dyn, default_update


def run_md(
    atoms,
    calculator,
    steps: int,
    temperature: float,
    timestep: float,
    *,
    mdalgo: int = 0,
    teend: float | None = None,
    smass: float | None = None,
    thermostat_params: Dict[str, float] | None = None,
    isif: int | None = 0,
    oszicar_pseudo_scf: bool = False,
    neb_mode: bool = False,
    neb_prev_positions: np.ndarray | None = None,
    neb_next_positions: np.ndarray | None = None,
    write_lammps_traj: bool = False,
    lammps_traj_interval: int = 1,
    lammps_traj_path: str = "lammps.lammpstrj",
):
    atoms.calc = _resolve_calculator(calculator)
    recorder = _initialize_vasp_compat_outputs(
        atoms,
        ibrion=0,
        isif=isif,
        potim=timestep,
        mdalgo=mdalgo,
        write_oszicar_pseudo_scf=oszicar_pseudo_scf,
        neb_mode=neb_mode,
        neb_prev_positions=neb_prev_positions,
        neb_next_positions=neb_next_positions,
    )
    if temperature <= 0:
        velocities = atoms.get_velocities()
        if velocities is None:
            zeros = [[0.0, 0.0, 0.0] for _ in range(len(atoms))]
            atoms.set_velocities(zeros)
        else:
            atoms.set_velocities(velocities * 0.0)
    else:
        velocitydistribution.MaxwellBoltzmannDistribution(
            atoms, temperature_K=temperature
        )
    params = thermostat_params or {}
    dyn, update_temperature = _select_md_dynamics(
        atoms,
        mdalgo,
        timestep,
        temperature,
        smass,
        params,
    )
    target_end = temperature if teend is None else teend
    md_step = 0

    def _log_md_state() -> None:
        nonlocal md_step
        md_step += 1
        potential_energy = atoms.get_potential_energy()
        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = 0.0
        thermostat_potential = _extract_numeric_attribute(
            dyn,
            (
                "thermostat_potential_energy",
                "thermostat_potential",
                "nose_potential_energy",
                "nhc_potential_energy",
            ),
        )
        thermostat_kinetic = _extract_numeric_attribute(
            dyn,
            (
                "thermostat_kinetic_energy",
                "thermostat_kinetic",
                "nose_kinetic_energy",
                "nhc_kinetic_energy",
            ),
        )
        total_energy = potential_energy + kinetic_energy + thermostat_potential + thermostat_kinetic
        try:
            temperature_inst = atoms.get_temperature()
        except Exception:
            temperature_inst = 0.0
        print(
            f"{md_step:7d} T={temperature_inst:7.1f} "
            f"E= {_format_energy_value(total_energy)} "
            f"F= {_format_energy_value(potential_energy)} "
            f"E0= {_format_energy_value(potential_energy)}  "
            f"EK= {_format_energy_value(kinetic_energy)} "
            f"SP= {_format_energy_value(thermostat_potential)} "
            f"SK= {_format_energy_value(thermostat_kinetic)}"
        )
        _record_vasp_compat_step(
            recorder,
            atoms,
            step_index=md_step,
            potential_energy=potential_energy,
            total_energy=total_energy,
            kinetic_energy=kinetic_energy,
            thermostat_potential=thermostat_potential,
            thermostat_kinetic=thermostat_kinetic,
            temperature=temperature_inst,
        )

    for i in range(steps):
        dyn.run(1)
        atoms.wrap()
        _log_md_state()
        _write_xdatcar_step("XDATCAR", atoms, i)
        if write_lammps_traj and i % lammps_traj_interval == 0:
            _write_lammps_trajectory_step(lammps_traj_path, atoms, i)
        if steps > 1 and i + 1 < steps and target_end != temperature:
            next_temp = temperature + (target_end - temperature) * (i + 1) / (steps - 1)
            update_temperature(next_temp)
    if not recorder.steps:
        potential_energy = atoms.get_potential_energy()
        kinetic_energy = 0.0
        try:
            kinetic_energy = atoms.get_kinetic_energy()
        except Exception:
            kinetic_energy = 0.0
        _record_vasp_compat_step(
            recorder,
            atoms,
            step_index=1,
            potential_energy=potential_energy,
            total_energy=potential_energy + kinetic_energy,
            kinetic_energy=kinetic_energy,
            temperature=float(temperature),
        )
    atoms.wrap()
    _write_vasprun_xml(recorder, atoms)
    _append_outcar_footer(recorder)
    write("CONTCAR", atoms, direct=True)
    return atoms.get_potential_energy()


def run_neb_images(
    *,
    workdir: str,
    incar,
    settings: IncarSettings,
    bcar: Dict[str, str],
    potcar_path: str | None,
    write_energy_csv: bool,
    write_lammps_traj: bool,
    lammps_traj_interval: int,
    oszicar_pseudo_scf: bool,
) -> None:
    """Run independent per-image calculations for a NEB-like directory layout."""

    workdir_abs = os.path.abspath(workdir)
    potcar_path_abs = os.path.abspath(potcar_path) if potcar_path else None
    pseudo_scf_settings = _pseudo_scf_settings_from_incar(incar, enabled=oszicar_pseudo_scf)
    input_paths = _VaspInputPaths(
        incar_path=os.path.join(workdir_abs, "INCAR"),
        potcar_path=potcar_path_abs or os.path.join(workdir_abs, "POTCAR"),
        kpoints_path=os.path.join(workdir_abs, "KPOINTS"),
    )
    image_dirs = _discover_neb_image_directories(workdir_abs)
    if len(image_dirs) < 2:
        raise RuntimeError(
            "NEB mode requires numbered image directories (for example 00, 01, 02)."
        )

    images_hint = _parse_neb_image_count(incar)
    if images_hint is not None:
        expected_dirs = images_hint + 2
        if expected_dirs != len(image_dirs):
            print(
                f"Warning: IMAGES={images_hint} implies {expected_dirs} image directories, "
                f"but found {len(image_dirs)} under {workdir_abs}. Proceeding with discovered directories."
            )

    with _active_pseudo_scf_settings(pseudo_scf_settings), _active_vasp_input_paths(input_paths):
        total_images = len(image_dirs)
        image_reference_positions: list[np.ndarray] = []
        for image_dir in image_dirs:
            structure_path = _resolve_neb_image_structure_path(image_dir)
            structure = read_structure(structure_path, potcar_path_abs)
            image_atoms = AseAtomsAdaptor.get_atoms(structure)
            image_atoms.wrap()
            image_reference_positions.append(np.asarray(image_atoms.get_positions(), dtype=float))

        for image_index, image_dir in enumerate(image_dirs, start=1):
            image_name = os.path.basename(image_dir)
            structure_path = _resolve_neb_image_structure_path(image_dir)
            structure = read_structure(structure_path, potcar_path_abs)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.wrap()
            _apply_initial_magnetization(atoms, incar)
            with _working_directory(workdir_abs):
                calculator = get_calculator(bcar, structure=structure)
            neb_prev_positions = image_reference_positions[image_index - 2] if image_index > 1 else None
            neb_next_positions = image_reference_positions[image_index] if image_index < total_images else None

            print(f"Running NEB image {image_name} ({image_index}/{total_images})")
            with _working_directory(image_dir):
                if settings.nsw <= 0 or settings.ibrion < 0:
                    run_single_point(
                        atoms,
                        calculator,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                    )
                elif settings.ibrion == 0:
                    run_md(
                        atoms,
                        calculator,
                        settings.nsw,
                        settings.tebeg,
                        settings.potim,
                        mdalgo=settings.mdalgo,
                        teend=settings.teend,
                        smass=settings.smass,
                        thermostat_params=settings.thermostat_params,
                        isif=settings.stress_isif,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                        write_lammps_traj=write_lammps_traj,
                        lammps_traj_interval=lammps_traj_interval,
                    )
                else:
                    run_relaxation(
                        atoms,
                        calculator,
                        settings.nsw,
                        settings.force_limit,
                        write_energy_csv,
                        isif=settings.isif,
                        pstress=settings.pstress,
                        energy_tolerance=settings.energy_tolerance,
                        ibrion=settings.ibrion,
                        stress_isif=settings.stress_isif,
                        neb_mode=True,
                        neb_prev_positions=neb_prev_positions,
                        neb_next_positions=neb_next_positions,
                        oszicar_pseudo_scf=oszicar_pseudo_scf,
                    )
        with _working_directory(workdir_abs):
            image_results = _collect_neb_image_results(image_dirs, potcar_path=potcar_path_abs)
            _write_neb_parent_aggregate_outputs(
                workdir=workdir_abs,
                settings=settings,
                image_results=image_results,
                oszicar_pseudo_scf=oszicar_pseudo_scf,
            )


def main():
    parser = argparse.ArgumentParser(description="Run MLP with VASP style inputs")
    parser.add_argument("--dir", default=".", help="Input directory")
    args = parser.parse_args()
    workdir = args.dir
    workdir_abs = os.path.abspath(workdir)

    poscar_path = os.path.join(workdir, "POSCAR")
    incar_path = os.path.join(workdir, "INCAR")
    potcar_path = os.path.join(workdir, "POTCAR")
    kpoints_path = os.path.join(workdir, "KPOINTS")
    bcar_path = os.path.join(workdir, "BCAR")

    for fname in ["KPOINTS", "WAVECAR", "CHGCAR"]:
        if os.path.exists(os.path.join(workdir, fname)):
            print(f"Note: {fname} detected but not used in MLP calculations.")

    incar = _load_incar(incar_path)
    bcar = parse_key_value_file(bcar_path) if os.path.exists(bcar_path) else {}

    write_energy_csv = _should_write_energy_csv(bcar)
    write_lammps_traj = _should_write_lammps_trajectory(bcar)
    write_pseudo_scf = _should_write_pseudo_scf(bcar)
    pseudo_scf_settings = _pseudo_scf_settings_from_incar(incar, enabled=write_pseudo_scf)
    _warn_for_unsupported_incar_tags(incar, pseudo_scf_enabled=write_pseudo_scf)
    settings = _load_incar_settings(incar)
    neb_mode = _is_neb_like_incar(incar)
    lammps_traj_interval = _get_lammps_trajectory_interval(bcar) if write_lammps_traj else 1
    potcar_for_structure = potcar_path if os.path.exists(potcar_path) else None
    input_paths = _VaspInputPaths(
        incar_path=os.path.abspath(incar_path),
        potcar_path=os.path.abspath(potcar_path),
        kpoints_path=os.path.abspath(kpoints_path),
    )

    with _active_pseudo_scf_settings(pseudo_scf_settings), _active_vasp_input_paths(input_paths):
        if neb_mode:
            neb_image_dirs = _discover_neb_image_directories(workdir)
            if neb_image_dirs:
                run_neb_images(
                    workdir=workdir,
                    incar=incar,
                    settings=settings,
                    bcar=bcar,
                    potcar_path=potcar_for_structure,
                    write_energy_csv=write_energy_csv,
                    write_lammps_traj=write_lammps_traj,
                    lammps_traj_interval=lammps_traj_interval,
                    oszicar_pseudo_scf=write_pseudo_scf,
                )
                print("Calculation completed.")
                return

        if not os.path.exists(poscar_path):
            if neb_mode:
                print(
                    "POSCAR not found. In NEB mode provide either a top-level POSCAR or "
                    "numbered image directories (00, 01, ...)."
                )
            else:
                print("POSCAR not found.")
            sys.exit(1)

        structure = read_structure(poscar_path, potcar_for_structure)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.wrap()
        _apply_initial_magnetization(atoms, incar)
        with _working_directory(workdir_abs):
            calculator = get_calculator(bcar, structure=structure)

        if settings.nsw <= 0 or settings.ibrion < 0:
            with _working_directory(workdir_abs):
                run_single_point(
                    atoms,
                    calculator,
                    isif=settings.stress_isif,
                    oszicar_pseudo_scf=write_pseudo_scf,
                )
        elif settings.ibrion == 0:
            with _working_directory(workdir_abs):
                run_md(
                    atoms,
                    calculator,
                    settings.nsw,
                    settings.tebeg,
                    settings.potim,
                    mdalgo=settings.mdalgo,
                    teend=settings.teend,
                    smass=settings.smass,
                    thermostat_params=settings.thermostat_params,
                    isif=settings.stress_isif,
                    oszicar_pseudo_scf=write_pseudo_scf,
                    write_lammps_traj=write_lammps_traj,
                    lammps_traj_interval=lammps_traj_interval,
                )
        else:
            with _working_directory(workdir_abs):
                run_relaxation(
                    atoms,
                    calculator,
                    settings.nsw,
                    settings.force_limit,
                    write_energy_csv,
                    isif=settings.isif,
                    pstress=settings.pstress,
                    energy_tolerance=settings.energy_tolerance,
                    ibrion=settings.ibrion,
                    stress_isif=settings.stress_isif,
                    neb_mode=neb_mode,
                    oszicar_pseudo_scf=write_pseudo_scf,
                )

    print("Calculation completed.")


if __name__ == "__main__":
    main()
