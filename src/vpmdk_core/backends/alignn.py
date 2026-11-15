"""ALIGNN-FF integration using the upstream ASE calculator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..backend_common import (
    ModelReferenceKind,
    _resolve_backend_model_reference,
    _resolve_device,
)


class _AlignnResultsMixin:
    """Restore ASE force dimensions and the energy used by ALIGNN gradients."""

    def calculate(
        self, atoms: Any, properties: Any = None, system_changes: Any = None
    ) -> None:
        # Upstream's ASE-to-JARVIS conversion drops PBC flags and always builds
        # periodic neighbors. Reject unsupported boundaries before any hooks
        # or upstream calculation can turn them into a different physical system.
        evaluation_atoms = atoms if atoms is not None else getattr(self, "atoms", None)
        if evaluation_atoms is not None and not np.all(evaluation_atoms.pbc):
            raise ValueError(
                "ALIGNN requires fully periodic structures (PBC enabled on all three axes); "
                "nonperiodic and partially periodic inputs are unsupported."
            )
        model = getattr(self, "model", None)
        config = getattr(model, "config", None)
        energy_mult_natoms = getattr(config, "energy_mult_natoms", True)
        # The pure model adds the penalty to a separate differentiation energy.
        # DGL models instead update `out` in place when energy_mult_natoms=False,
        # so that case already includes the penalty in the reported energy.
        missing_penalty = getattr(config, "use_penalty", False) and (
            energy_mult_natoms or config.name == "alignn_atomwise_pure"
        )
        penalty_energy = 0.0

        def capture_penalty(module: Any, inputs: Any, output: Any) -> None:
            nonlocal penalty_energy
            # Use the actual inference graph, retaining upstream's directed
            # edges, periodic images and neighbor truncation. Detaching keeps
            # the pretrained force/stress differentiation entirely untouched.
            graph = inputs[0][0]
            lengths = graph.edata["r"].detach().norm(dim=1)
            if (
                config.name == "alignn_atomwise"
                and config.use_cutoff_function
                and not config.multiply_cutoff
            ):
                # DGL overwrites bondlength before forming the penalty. Pure
                # PyTorch keeps its transformed embedding distances separate.
                from alignn.models.utils import cutoff_function_based_edges

                lengths = cutoff_function_based_edges(
                    lengths, inner_cutoff=config.inner_cutoff, exponent=config.exponent
                )
            penalty_energy = (
                (config.penalty_threshold - lengths).clamp_min(0)
                * config.penalty_factor
            ).sum().item()
            if energy_mult_natoms:
                penalty_energy /= len(self.atoms)
            if self.intensive:
                penalty_energy *= len(self.atoms)

        handle = model.register_forward_hook(capture_penalty) if missing_penalty else None
        try:
            super().calculate(atoms, properties, system_changes)
        finally:
            if handle is not None:
                handle.remove()
        if missing_penalty:
            self.results["energy"] += penalty_energy
        if "forces" in self.results:
            self.results["forces"] = np.asarray(self.results["forces"]).reshape(
                len(self.atoms), 3
            )


def _get_alignn_calculator_cls() -> Any:
    """Import lazily so ALIGNN remains an optional dependency."""
    try:
        from alignn.ff.calculators import AlignnAtomwiseCalculator
    except ImportError:
        return None

    class VPMDKAlignnCalculator(_AlignnResultsMixin, AlignnAtomwiseCalculator):
        """Upstream ALIGNN calculator with consistent energy and ASE forces."""

    return VPMDKAlignnCalculator


def _alignn_model_files(model_path: str | Path) -> tuple[Path, Path]:
    """Keep the lexical weights parent when selecting the sibling config."""
    selected = Path(model_path)
    checkpoint = selected / "best_model.pt" if selected.is_dir() else selected
    config_path = checkpoint.parent / "config.json"
    for required in (checkpoint, config_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"ALIGNN requires a checkpoint and config.json: {required}"
            )
    return checkpoint, config_path


def _alignn_numeric_setting(model: dict[str, Any], key: str, default: float) -> float:
    """Do not let implicit coercion hide a nonstandard gradient coefficient."""
    value = model.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"ALIGNN {key} must be a finite JSON number")
    return float(value)


def _validate_alignn_model_convention(model: dict[str, Any]) -> None:
    """Accept only the tested scalar E/F/S inference paths and conventions."""
    name = model.get("name")
    if name not in {"alignn_atomwise", "alignn_atomwise_pure"}:
        raise ValueError(
            f"Unsupported ALIGNN force field architecture {name!r}. "
            "Only alignn_atomwise and alignn_atomwise_pure have validated E/F/S paths; "
            "ealignn_atomwise torque postprocessing is not supported."
        )
    flags: dict[str, bool] = {}
    for key, default in (
        ("energy_mult_natoms", True),
        ("force_mult_natoms", False),
        ("include_pos_deriv", False),
        ("add_reverse_forces", True),
        ("lg_on_fly", True),
        ("classification", False),
        ("calculate_gradient", False),
    ):
        value = model.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(
                f"ALIGNN force normalization flag {key} must be a JSON boolean"
            )
        flags[key] = value

    if not flags["calculate_gradient"] or flags["classification"]:
        raise ValueError(
            "ALIGNN force field requires calculate_gradient=true and classification=false"
        )
    output_features = model.get("output_features", 1)
    if type(output_features) is not int or output_features != 1:
        raise ValueError("ALIGNN force field requires output_features=1")
    layers = model.get("alignn_layers", 2)
    if type(layers) is not int or layers < 0:
        raise ValueError("ALIGNN alignn_layers must be a nonnegative JSON integer")
    if name == "alignn_atomwise_pure" and layers == 0:
        raise ValueError(
            "ALIGNN pure ASE inference requires alignn_layers>0 (three graph inputs)"
        )
    if layers > 0 and not flags["lg_on_fly"]:
        raise ValueError(
            "ALIGNN alignn_layers>0 requires lg_on_fly=true to differentiate bond angles"
        )
    if _alignn_numeric_setting(model, "grad_multiplier", -1) != -1:
        raise ValueError("ALIGNN E/F/S requires grad_multiplier=-1")
    if _alignn_numeric_setting(model, "stress_multiplier", 1) != 1:
        raise ValueError("ALIGNN E/F/S requires stress_multiplier=1")
    if model.get("link", "identity") != "identity":
        raise ValueError(
            "ALIGNN force field requires link=identity; "
            "nonlinear output links omit gradient terms"
        )
    if _alignn_numeric_setting(model, "stresswise_weight", 0) < 0:
        raise ValueError("ALIGNN stresswise_weight must be nonnegative")

    cartesian_derivatives = name == "alignn_atomwise" and flags["include_pos_deriv"]
    # Pure PyTorch ignores include_pos_deriv. DGL Cartesian derivatives already
    # include both bond ends but do not provide usable stress.
    uses_edge_derivatives = name == "alignn_atomwise_pure" or (
        name == "alignn_atomwise" and not cartesian_derivatives
    )
    if uses_edge_derivatives and not flags["add_reverse_forces"]:
        raise ValueError(
            "ALIGNN MODEL omits outgoing-edge force contributions "
            "(add_reverse_forces=false), so forces do not match energy gradients. "
            "Use a validated checkpoint such as v12.2.2024_dft_3d_307k or "
            "alignn_ff_mptrj; changing pretrained force conventions is not supported."
        )

    # ASE intensive=True returns N*out. Edge derivatives multiply by N once
    # for each enabled atom-count flag. Cartesian DGL instead multiplies by N
    # unconditionally and ignores force_mult_natoms.
    force_power = int(flags["energy_mult_natoms"]) + (
        1 if cartesian_derivatives else int(flags["force_mult_natoms"])
    )
    if force_power != 1:
        settings = ", ".join(
            f"{key}={str(value).lower()}" for key, value in flags.items()
        )
        raise ValueError(
            f"ALIGNN MODEL has inconsistent force normalization ({name}: {settings}): "
            f"forces scale with N^{force_power}, but reported total-energy gradients "
            "scale with N. Use a validated checkpoint such as "
            "v12.2.2024_dft_3d_307k or alignn_ff_mptrj; "
            "changing pretrained gradient scales is not supported."
        )
    # Upstream computes stress only in the edge-derivative branch. Cartesian
    # differentiation leaves a placeholder which its ASE wrapper turns into
    # zeros. This backend always enables stress, so do not accept that branch.
    if cartesian_derivatives:
        raise ValueError(
            "ALIGNN DGL Cartesian derivatives have no usable stress "
            "(alignn_atomwise, include_pos_deriv=true). "
            "VPMDK requires energy, forces and stress from this backend. "
            "Use a validated checkpoint such as v12.2.2024_dft_3d_307k or "
            "alignn_ff_mptrj with its original configuration."
        )
    # Pure ZBL is always returned as total energy, but its differentiated term
    # also receives force_mult_natoms. The alternate (false,true) convention is
    # therefore compatible only when ZBL is explicitly disabled.
    if (
        name == "alignn_atomwise_pure"
        and flags["force_mult_natoms"]
        and model.get("use_zbl") is not False
    ):
        raise ValueError(
            "ALIGNN MODEL has inconsistent ZBL energy/force normalization: "
            "force_mult_natoms=true multiplies ZBL forces but not its total energy. "
            "Use a checkpoint with a consistent pretrained normalization."
        )


def _build_alignn_calculator(bcar_tags: dict[str, str], *, structure=None) -> Any:
    """Load a named force field or a checkpoint with a sibling config.json."""
    reference = _resolve_backend_model_reference("ALIGNN", bcar_tags.get("MODEL"))
    calculator_cls = _get_alignn_calculator_cls()
    if calculator_cls is None:
        raise RuntimeError(
            "ALIGNN-FF calculator not available. Install alignn and its dependencies; "
            "legacy checkpoints also require a compatible DGL installation."
        )
    if reference.kind is ModelReferenceKind.LOCAL_PATH:
        checkpoint, config_path = _alignn_model_files(reference.value)
    else:
        from alignn.ff.calculators import get_all_models, get_figshare_model_ff

        name = reference.value
        if name not in get_all_models():
            raise ValueError(f"Unsupported ALIGNN force-field MODEL {name!r}")
        downloaded = get_figshare_model_ff(model_name=name)
        if not downloaded:
            raise RuntimeError(f"Unable to download ALIGNN MODEL {name!r}")
        checkpoint, config_path = _alignn_model_files(downloaded)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    model = config.get("model", {})
    _validate_alignn_model_convention(model)
    # stresswise_weight only gates inference stress; it does not rescale it.
    # Upstream's include_stress=True misses omitted keys. Enable this gate in
    # our private configuration, without changing files or pretrained forces.
    runtime_model = dict(model)
    # The model schema supplies this default, but the upstream ASE calculator
    # indexes the raw configuration directly when choosing its graph inputs.
    runtime_model.setdefault("alignn_layers", 2)
    if runtime_model.get("stresswise_weight", 0) == 0:
        runtime_model["stresswise_weight"] = 0.1
    runtime_config = {**config, "model": runtime_model}
    return calculator_cls(
        path=str(checkpoint.parent),
        model_filename=checkpoint.name,
        device=_resolve_device(bcar_tags.get("DEVICE")) or "cpu",
        include_stress=True,
        intensive=True,
        config=runtime_config,
        batch_stress=True,
        force_multiplier=1,
        force_mult_natoms=False,
        force_mult_batchsize=False,
        stress_wt=1.0,
    )
