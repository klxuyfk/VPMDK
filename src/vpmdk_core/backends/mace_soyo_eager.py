"""Direct checkpoint inference using MACE-SOYO's tensor E/F/S implementation."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .mace_soyo_common import _cueq_checkpoint_compat, _naive_segmented_polynomials


class MaceSoyoEagerCalculator(Calculator):
    """Evaluate an unconditioned SOYO checkpoint on CUDA without AOTI."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model_path: str, *, device: str, head: str | None,
                 cueq_method: str = "native") -> None:
        super().__init__()
        import torch
        import cuequivariance_torch as cuet
        from mace_soyo.utils.device import resolve_cuda_device
        from mace_soyo.utils.model_checkpoint import load_model_from_checkpoint
        from mace_soyo.utils.compile_utils import TensorEFS
        from mace_soyo.inference.ase_calc import MACESoyoCalculator, get_neighbor_list_gpu_pbc

        self.device = resolve_cuda_device(device)
        self.dtype = torch.float32
        self.cueq_method = cueq_method
        context = (_naive_segmented_polynomials(cuet.SegmentedPolynomial)
                   if cueq_method == "naive" else nullcontext())
        with _cueq_checkpoint_compat(), context:
            self.model = load_model_from_checkpoint(model_path, self.device, self.dtype)
        if self.model.use_spin_charge:
            raise ValueError("Direct MACE-SOYO checkpoints require an unconditioned model; export conditioned models to .pt2.")
        self.head_names = list(self.model.head_names)
        if head is None:
            if len(self.head_names) != 1:
                raise ValueError("Set MACE_SOYO_HEAD for a multi-head checkpoint.")
            head = self.head_names[0]
        if head not in self.head_names:
            raise ValueError(f"Unknown MACE_SOYO_HEAD {head!r}; available: {self.head_names}")
        self.head = head
        self.cutoff = self.model.cutoff
        self._head_id = torch.tensor([self.head_names.index(head)], device=self.device, dtype=torch.long)
        self._wrapper = TensorEFS(self.model)
        self._neighbor_list_fn = get_neighbor_list_gpu_pbc
        self._input_builder = MACESoyoCalculator._atoms_to_inputs

    def calculate(self, atoms: Any = None, properties: Any = None,
                  system_changes: Any = all_changes) -> None:
        import torch

        super().calculate(atoms, properties, system_changes)
        z, frac, cell, batch, edges, images = self._input_builder(self, self.atoms)
        pbc = torch.tensor([self.atoms.pbc.tolist()], device=self.device, dtype=torch.bool)
        with torch.cuda.device(self.device), torch.enable_grad():
            energy, forces, stress = self._wrapper(
                z, frac, cell.unsqueeze(0), batch, edges, images, self._head_id, pbc
            )
        value = float(energy.detach().cpu().item())
        self.results = {
            "energy": value, "free_energy": value,
            "forces": np.asarray(forces.detach().cpu().numpy(), dtype=np.float64),
            "stress": full_3x3_to_voigt_6_stress(stress[0].detach().cpu().numpy()).astype(np.float64),
        }
